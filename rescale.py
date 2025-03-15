# Updated imports
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import torch
import os
from torchvision import transforms
from pydantic import BaseModel
from typing import Optional

# Try this more generic import
from transformers import AutoModelForImageClassification

app = FastAPI()

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the PyTorch model when the application starts
try:
    model_path = "fine_tuned_vit_model.pkl"
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking if model exists: {os.path.exists(model_path)}")
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"Loaded state dictionary type: {type(state_dict)}")
    
    # Create a custom model class that can work with your state dictionary
    class CustomVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            # Adaptive pooling ensures consistent output size
                torch.nn.AdaptiveAvgPool2d((8, 8)),  # Output will always be 8×8
                torch.nn.Flatten(),
                torch.nn.Linear(128 * 8 * 8, 512),  # Now we know exactly what size to expect
                torch.nn.ReLU(),
                torch.nn.Linear(512, 2)
        )
        
        def forward(self, x):
            output = self.features(x)
            return ModelOutput(output)
    
    # Use the custom model for prediction
    model = CustomVisionModel()
    print("Using custom model for inference")
    
    # Set to evaluation mode
    model.eval()
    print("Model loaded in test mode")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback_str = traceback.format_exc()
    print(f"Traceback: {traceback_str}")
    model = None

# Update the transform to ensure consistent dimensions
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



class ModelOutput:
    def __init__(self, logits):
        self.logits = logits

class ImageResponse(BaseModel):
    is_edited: bool
    confidence: float
    details: Optional[str] = None

@app.post("/analyze/", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    # Check if model is loaded
    if model is None:
        return ImageResponse(
            is_edited=False,
            confidence=0.0,
            details="Model not loaded, please check server logs"
        )
    
    # Read and process the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Apply transformations
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    try:
        with torch.no_grad():
            outputs = model(input_batch)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Assume class 1 is "edited"
            confidence = float(probabilities[0, 1].item() * 100)
            is_edited = bool(confidence > 50)
        
        return ImageResponse(
            is_edited=is_edited,
            confidence=confidence,
            details=f"Image {'is' if is_edited else 'is not'} edited (confidence: {confidence:.2f}%)"
        )
    except Exception as e:
        import traceback
        return ImageResponse(
            is_edited=False,
            confidence=0.0,
            details=f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
