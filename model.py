import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class AdaptiveActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(dim) * 0.01)
        self.beta = nn.Parameter(torch.ones(dim))
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
    
    def forward(self, x):
        gate = torch.sigmoid(self.alpha)
        return self.beta * (gate * self.elu(x) + (1 - gate) * self.gelu(x))

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        return out

class CrossDimTransform(nn.Module):
    def __init__(self, dim, hidden_mult=2):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        transform = self.net(x)
        gate = self.gate(x)
        return x + gate * transform

class ACTBlock(nn.Module):
    def __init__(self, dim, ff_mult=2, use_attention=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(dim)
        
        self.fc1 = nn.Linear(dim, dim * ff_mult)
        self.act = AdaptiveActivation(dim * ff_mult)
        self.cross = CrossDimTransform(dim * ff_mult, hidden_mult=0.5)
        self.fc2 = nn.Linear(dim * ff_mult, dim)
        self.dropout = nn.Dropout(0.15)
        
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        residual = x
        
        if self.use_attention:
            x = self.norm1(x)
            x = x + self.attention(x)
        
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.cross(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return residual + self.residual_scale * x

class ACTNet(nn.Module):
    def __init__(self, cont_dim=3, cat_cardinalities=[3], emb_dim=16, hidden_dim=256, 
                 n_blocks=4, num_classes=3, dropout=0.2):
        """
        Updated for RSU Offloading Inference:
        cont_dim: bandwidth, critical_task, mips (3)
        cat_cardinalities: mobility_status (3: Low, Medium, High)
        num_classes: local, cloud, rsu (3)
        """
        super().__init__()
        
        self.embs = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(card, emb_dim),
                nn.Dropout(dropout * 0.5)
            ) for card in cat_cardinalities
        ])
        
        emb_total = emb_dim * len(cat_cardinalities) if len(cat_cardinalities) > 0 else 0
        input_dim = cont_dim + emb_total
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.feature_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            ACTBlock(hidden_dim, use_attention=(i % 2 == 0))
            for i in range(n_blocks)
        ])
        
        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, cont, cat):
        embs = []
        if cat is not None and cat.size(1) > 0:
            for i, emb_layer in enumerate(self.embs):
                # Force indices to Long for embedding lookup
                cat_indices = cat[:, i].long() 
                embs.append(emb_layer(cat_indices))
            embs = torch.cat(embs, dim=1)
            x = torch.cat([cont, embs], dim=1)
        else:
            x = cont
        
        x = self.input_proj(x)
        x = x + self.feature_interaction(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.pre_head_norm(x)
        logits = self.class_head(x)
        
        return logits

# --- LOSS FUNCTIONS REMAIN UNCHANGED FOR TRAINING ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = (-smooth_one_hot * log_prob).sum(dim=1).mean()
        return loss