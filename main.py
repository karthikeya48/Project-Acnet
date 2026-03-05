from typing import Dict
import torch
import joblib
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import ACTNet 
from hato_rsu_selection import hato_rsu_selector  
app = FastAPI(title="ACTNet + HATO Task Offloading API")

# --- 1. GLOBAL LOAD (Persistent in memory) ---
try:
    # Match your training: 3 continuous features, 1 categorical (3 levels), 3 output classes
    model = ACTNet(cont_dim=3, cat_cardinalities=[3], num_classes=3)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # Load Pickled Preprocessors
    scaler = joblib.load("scaler.pkl")
    le_mobility = joblib.load("le_mobility.pkl")
    le_target = joblib.load("le_target.pkl")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

weight_storage: Dict[str, dict] = {}


# --- 2. INPUT SCHEMA ---
class OffloadRequest(BaseModel):
    bandwidth_mbps: float
    critical_task: int
    mobility_status: str
    number_of_instructions_mips: float
    Infrastructure: dict

# class TrainingUpdate(TaskData):
#     actual_target: int  



# --- . ENDPOINT ---
@app.post("/predict")
async def predict_offload_decision(request: OffloadRequest):
    try:
        # A. Preprocessing
        # Reshape for scaler: [[bandwidth, critical, mips]]
        raw_cont = [[request.bandwidth_mbps, request.critical_task, request.number_of_instructions_mips]]
        scaled_cont = scaler.transform(raw_cont)
        
        # Map string (e.g., "High") to integer index
        cat_val = le_mobility.transform([request.mobility_status])
        
        # B. Model Inference
        with torch.no_grad():
            t_cont = torch.tensor(scaled_cont, dtype=torch.float32)
            t_cat = torch.tensor([cat_val], dtype=torch.long)
            logits = model(t_cont, t_cat)
            class_idx = torch.argmax(logits, dim=1).item()
        
        # C. Decode high-level decision (Local, Cloud, RSU)
        decision = le_target.inverse_transform([class_idx])[0]
        
        # D. HATO Layer (If ACTNet says RSU, we pick the specific one)
        final_target = decision
        if decision.lower() == "rsu":
            rsu_pool = request.Infrastructure.get("Rsu", {})
            if rsu_pool:
                final_target = hato_rsu_selector(request.number_of_instructions_mips, rsu_pool)
            else:
                final_target = "Generic RSU (No pool provided)"

        return {
            "actnet_decision": decision,
            "target_node": final_target,
            "input_summary": {
                "mips": request.number_of_instructions_mips,
                "mobility": request.mobility_status
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/update_local")
# async def update_local(update: TrainingUpdate):
#     local_model = copy.deepcopy(global_model)
#     local_model.train()
#     optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

#     numeric = [update.signal_strength, update.critical_task, update.bandwidth_mbps, 
#                update.number_of_instructions, update.task_size_MB]
#     numeric_scaled = scaler.transform(np.array(numeric).reshape(1, -1)).astype(np.float32)
#     cont_t = torch.from_numpy(numeric_scaled).to(device)

#     le = cat_encoders['mobility_status']
#     mob_encoded = le.transform(np.array([update.mobility_status]))
#     cat_t = torch.from_numpy(mob_encoded).reshape(1, -1).to(device)

#     label_t = torch.LongTensor([update.actual_target]).to(device)

#     # 4. Train 1 Step
#     optimizer.zero_grad()
#     logits = local_model(cont_t, cat_t)
#     loss = criterion(logits, label_t)
#     loss.backward()
#     optimizer.step()

#     weight_storage[update.vehicle_id] = local_model.state_dict()

#     print(f"Stored local weights for {update.vehicle_id}")
    
#     return {"message": "Local update successful"}
# @app.post("/aggregate")
# async def aggregate():
#     """Cloud triggers this to average all weights and update Global Model."""
#     if not weight_storage:
#         raise HTTPException(status_code=400, detail="No vehicle weights available.")
#     all_weights = list(weight_storage.values())
#     avg_weights = copy.deepcopy(all_weights[0])
    
#     for key in avg_weights.keys():
#         for i in range(1, len(all_weights)):
#             avg_weights[key] += all_weights[i][key]
#         avg_weights[key] = torch.div(avg_weights[key], len(all_weights))
#     global_model.load_state_dict(avg_weights)
#     torch.save(global_model.state_dict(), "Global.pth")
#     weight_storage.clear() 
#     print("Global model updated via Federated Averaging.")
#     return {"message": "Global Model Updated successfully via Federated Averaging."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)