from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from land_change import detect_land_changes
from species_mont import SpeciesMonitoringSystem
from threat import WildlifeDetectionSystem

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
import time


print("🔄 Loading models... Please wait.")
start_time = time.time()
species_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
species_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").eval()
yolo_model = YOLO("yolov8x.pt")
threat_model = pipeline("image-classification", model="nateraw/vit-base-beans")


end_time = time.time()
print(f"✅ Models loaded in {end_time - start_time:.2f} seconds!")
species_monitor = SpeciesMonitoringSystem()
threat_detection = WildlifeDetectionSystem()

@app.post("/detect_species/")
async def detect_species(file: UploadFile = File(...)):
    image = Image.open(file.file)
    results = species_monitor.detect_species(image)
    return {"species_detected": results}

@app.post("/count_population/")
async def count_population(file: UploadFile = File(...)):
    image = Image.open(file.file)
    count, _ = species_monitor.count_population(image)
    return {"count": count}

@app.post("/assess_health/")
async def assess_health(file: UploadFile = File(...)):
    image = Image.open(file.file)
    status, score, indicators = species_monitor.assess_health(image)
    return {"status": status, "score": score, "indicators": indicators}

@app.post("/detect_land_changes/")
async def land_changes(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    changes = detect_land_changes(file1.file, file2.file)
    return {"land_change_detected": True if changes is not None else False}

@app.post("/detect_threat/")
async def detect_threat(file: UploadFile = File(...)):
    image = Image.open(file.file)
    results, _ = threat_detection.detect_poaching(image)
    return {"threats_detected": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
