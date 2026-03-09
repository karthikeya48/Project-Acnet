import torch
import torch.nn as nn
import joblib
import copy
import uvicorn
from typing import Dict
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the model architecture from model.py
from model import ACTNet
from hato_rsu_selection import hato_rsu_selector 

app = FastAPI(title="ACTNet + FedAvg Offloading API")

# --- GLOBAL STATE ---
lock = Lock()
weight_storage: Dict[str, dict] = {}
global_model = None
scaler = None
le_mobility = None
le_target = None

# --- INPUT SCHEMA ---
class OffloadRequest(BaseModel):
    bandwidth_mbps: float
    critical_task: int
    mobility_status: str
    number_of_instructions_mips: float
    Infrastructure: dict

class TrainingUpdate(OffloadRequest):
    vehicle_id: str
    actual_target: str 

# --- INITIALIZATION ---
@app.on_event("startup")
def load_artifacts():
    global global_model, scaler, le_mobility, le_target
    try:
        global_model = ACTNet(cont_dim=3, cat_cardinalities=[3], num_classes=3)
        global_model.load_state_dict(torch.load("model.pth", map_location='cpu', weights_only=True))
        global_model.eval()
        
        scaler = joblib.load("scaler.pkl")
        le_mobility = joblib.load("le_mobility.pkl")
        le_target = joblib.load("le_target.pkl")
        print("Model and preprocessors loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

# --- API ENDPOINTS ---

@app.post("/predict")
async def predict_decision(request: OffloadRequest):
    try:
        # Preprocessing
        raw_cont = [[request.bandwidth_mbps, request.critical_task, request.number_of_instructions_mips]]
        scaled_cont = torch.tensor(scaler.transform(raw_cont), dtype=torch.float32)
        cat_idx = torch.tensor([le_mobility.transform([request.mobility_status])], dtype=torch.long)

        # Inference
        with torch.no_grad():
            logits = global_model(scaled_cont, cat_idx)
            class_idx = torch.argmax(logits, dim=1).item()
        
        decision = le_target.inverse_transform([class_idx])[0]
        final_target = decision

        # HATO Layer for RSU selection
        if decision.lower() == "rsu":
            rsu_pool = request.Infrastructure.get("Rsu", {})
            if rsu_pool:
                final_target = hato_rsu_selector(request.number_of_instructions_mips, rsu_pool)
            else:
                final_target = "Generic RSU (Pool Empty)"

        return {
            "actnet_decision": decision,
            "target_node": final_target
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/local_update")
async def local_update(update: TrainingUpdate):
    """
    Simulates a vehicle performing local training and sharing its model weights.
    """
    try:
        # Clone global model for local fine-tuning
        local_model = copy.deepcopy(global_model)
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Prepare Tensors
        raw_cont = [[update.bandwidth_mbps, update.critical_task, update.number_of_instructions_mips]]
        scaled_cont = torch.tensor(scaler.transform(raw_cont), dtype=torch.float32)
        cat_idx = torch.tensor([le_mobility.transform([update.mobility_status])], dtype=torch.long)
        
        # Encode the 'correct' answer sent by the vehicle
        target_idx = torch.tensor([le_target.transform([update.actual_target])[0]], dtype=torch.long)

        # Optimization step
        optimizer.zero_grad()
        output = local_model(scaled_cont, cat_idx)
        loss = criterion(output, target_idx)
        loss.backward()
        optimizer.step()

        # Save weights to server-side buffer
        with lock:
            weight_storage[update.vehicle_id] = local_model.state_dict()
        
        return {"status": "Weights received", "local_loss": float(loss)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/aggregate")
async def aggregate_federated_model():
    """
    Combines all collected vehicle weights using Federated Averaging (FedAvg).
    """
    with lock:
        if len(weight_storage) < 1:
            raise HTTPException(status_code=400, detail="No updates available for aggregation.")
        
        all_updates = list(weight_storage.values())
        avg_weights = copy.deepcopy(all_updates[0])

        # FedAvg: Calculate mean of parameters
        for key in avg_weights.keys():
            for i in range(1, len(all_updates)):
                avg_weights[key] += all_updates[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(all_updates))

        # Update Global Model
        global_model.load_state_dict(avg_weights)
        torch.save(global_model.state_dict(), "model.pth")
        
        # Clear buffer for next round
        num_clients = len(weight_storage)
        weight_storage.clear()
        
    return {"message": f"Global model updated using updates from {num_clients} clients."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)