import pandas as pd
import torch
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import ACTNet, FocalLoss  

# 1. Load your CSV
df = pd.read_csv("Global.csv")

df = df[df['offloading_target'] != 'peer_vehicle']
print(df.head())  

# 2. Setup Preprocessing
# Mobility: High, Medium, Low -> 0, 1, 2
le_mobility = LabelEncoder()
df['mobility_status'] = le_mobility.fit_transform(df['mobility_status'])

# Target: local, cloud, rsu -> 0, 1, 2
le_target = LabelEncoder()
df['offloading_target'] = le_target.fit_transform(df['offloading_target'].str.lower())

# Continuous Scaling
cont_cols = ['bandwidth_mbps', 'critical_task', 'number_of_instructions_mips']
scaler = StandardScaler()
df[cont_cols] = scaler.fit_transform(df[cont_cols])

# 3. Convert to Tensors
cont_tensor = torch.tensor(df[cont_cols].values, dtype=torch.float32)
cat_tensor = torch.tensor(df[['mobility_status']].values, dtype=torch.long)
target_tensor = torch.tensor(df['offloading_target'].values, dtype=torch.long)

# 4. Initialize ACTNet
# cont_dim = 3 (bandwidth, task, mips)
# cat_cardinalities = [3] (Low, Med, High)
# num_classes = 3 (Local, Cloud, RSU)
model = ACTNet(cont_dim=3, cat_cardinalities=[3], num_classes=3)
criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 5. Simple Training Loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(cont_tensor, cat_tensor)
    loss = criterion(outputs, target_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 6. THE IMPORTANT PART: SAVING
# Save the model weights
torch.save(model.state_dict(), "model.pth")

# Save the preprocessing tools for FastAPI
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_mobility, "le_mobility.pkl")
joblib.dump(le_target, "le_target.pkl")

print("Files generated: model.pth, scaler.pkl, le_mobility.pkl, le_target.pkl")