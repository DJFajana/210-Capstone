import os
import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import the CORS middleware
from pydantic import BaseModel
from torchvision import transforms
from transformers import ViTForImageClassification
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Edit Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://rufakeapp.com"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class ImageResponse(BaseModel):
    is_edited: bool
    confidence: float

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Global variables
model = None
device = None

def get_device():
    """Determine the device to use based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        logger.warning("CUDA is not available, using CPU instead")
        return "cpu"

def load_model():
    """Load the ViT model"""
    global model, device
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Check for model file
    model_path = "allpurdue_vit_model.pkl"
    
    try:
        # Initialize the ViT model
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights if available
        if os.path.exists(model_path):
            logger.info(f"Loading weights from {model_path}")
            # Load state dict and handle CUDA vs CPU
            state_dict = torch.load(model_path, map_location=device)
            
            try:
                model.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading state dict directly: {e}")
                logger.warning("Using pretrained model only")
        else:
            logger.warning(f"No model file found at {model_path}")
            logger.warning("Using pretrained model only")
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def predict_image(image):
    """Run prediction on an image"""
    global model, device
    
    # If model is not loaded, load it
    if model is None:
        success = load_model()
        if not success:
            logger.error("Failed to load model")
            return None, 0.0
    
    try:
        # Process the image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor).logits
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            is_edited = prediction.item() == 1  # Assuming class 1 is "edited"
            confidence_score = confidence.item()
        
        return is_edited, confidence_score
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 0.0

@app.post("/analyze/", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an image for potential editing"""
    try:
        # Read and process the image
        image_data = await file.read()
        
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get prediction
        result = predict_image(image)
        
        # If prediction failed, return an error
        if result is None:
            logger.error("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        is_edited, confidence = result
        
        return ImageResponse(
            is_edited=is_edited,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model, device
    
    # Check CUDA status
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_name = torch.cuda.get_device_name(0)
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not set",
        "cuda_available": cuda_available,
        "device_name": device_name
    }

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
