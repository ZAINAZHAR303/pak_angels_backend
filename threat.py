import streamlit as st
import torch
import torchvision
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
import yaml
from pathlib import Path
import requests
from io import BytesIO

class WildlifeDetectionSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        
    def load_models(self):
        # Load MobileNetV3 for general classification
        self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet.eval()
        self.mobilenet.to(self.device)
        
        # Load YOLOv5 for object detection
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.yolo.eval()
        self.yolo.to(self.device)
        
        # Transform pipeline for MobileNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load ImageNet labels
        self.labels = requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json').json()
        
        # Custom labels for each detection type
        self.custom_labels = {
            'poaching': ['person', 'truck', 'car', 'motorcycle', 'boat'],
            'invasive': ['snake', 'fish', 'boar', 'bird'],
            'environmental': ['fire', 'smoke', 'flood', 'landslide']
        }

    def preprocess_image(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image format")
        return image

    def detect_poaching(self, image):
        # Use YOLOv5 for poaching detection
        image_np = np.array(image)
        results = self.yolo(image_np)
        
        detections = {}
        for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
            label = results.names[int(cls)]
            if label in self.custom_labels['poaching']:
                detections[label] = float(conf)
                
        return detections, results

    def detect_invasive_species(self, image):
        # Use MobileNet for species classification
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.mobilenet(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        results = {}
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = self.labels[idx]
            if any(invasive in label.lower() for invasive in self.custom_labels['invasive']):
                results[label] = float(prob)
                
        return results

    def detect_environmental_hazards(self, image):
        # Use both models for environmental hazard detection
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # MobileNet for classification
            output = self.mobilenet(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # YOLO for object detection
        image_np = np.array(image)
        yolo_results = self.yolo(image_np)
        
        results = {}
        
        # Combine results from both models
        for hazard in self.custom_labels['environmental']:
            confidence = max(
                [float(conf) for *_, conf, cls in yolo_results.xyxy[0] 
                 if yolo_results.names[int(cls)].lower() == hazard],
                default=0.0
            )
            results[hazard] = confidence
            
        return results, yolo_results

# Streamlit UI
# st.set_page_config(page_title="Advanced Wildlife Threat Detection", layout="wide")

# @st.cache_resource
# def load_detection_system():
#     return WildlifeDetectionSystem()

# system = load_detection_system()

# st.title("🦁 Advanced Wildlife Threat Detection System")

# # Sidebar navigation
# page = st.sidebar.radio("Select System:", 
#     ["Poaching Detection", "Invasive Species", "Environmental Hazards"])

# # File uploader
# uploaded_file7 = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# if uploaded_file7:
#     image = Image.open(uploaded_file7)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     if page == "Poaching Detection":
#         st.subheader("🎯 Poaching Activity Detection")
        
#         with st.spinner("Analyzing image for potential poaching activities..."):
#             detections, results = system.detect_poaching(image)
            
#             # Display results
#             if detections:
#                 for obj, conf in detections.items():
#                     st.progress(conf)
#                     st.write(f"{obj.title()}: {conf*100:.1f}% confidence")
                
#                 # Display annotated image
#                 results.render()
#                 st.image(results.ims[0], caption="Detected Objects", use_column_width=True)
                
#                 if any(conf > 0.7 for conf in detections.values()):
#                     st.error("⚠️ High-risk poaching activity detected! Alert sent to authorities.")
#             else:
#                 st.success("No suspicious activities detected.")
                
#     elif page == "Invasive Species":
#         st.subheader("🦈 Invasive Species Detection")
        
#         with st.spinner("Analyzing image for invasive species..."):
#             detections = system.detect_invasive_species(image)
            
#             if detections:
#                 for species, conf in detections.items():
#                     st.progress(conf)
#                     st.write(f"{species.title()}: {conf*100:.1f}% confidence")
                
#                 # Display historical data
#                 st.subheader("Historical Sightings")
#                 dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
#                 data = pd.DataFrame({
#                     'Date': dates,
#                     'Sightings': np.random.randint(1, 10, size=10)
#                 })
#                 fig = px.line(data, x='Date', y='Sightings', 
#                              title='Invasive Species Sightings Trend')
#                 st.plotly_chart(fig)
#             else:
#                 st.success("No invasive species detected.")
                
    

# # Display system status
# st.sidebar.markdown("---")
# st.sidebar.markdown("### System Status")
# st.sidebar.markdown(f"🖥️ Using device: {system.device}")
# st.sidebar.markdown("✅ Models loaded successfully")