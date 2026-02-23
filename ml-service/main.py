"""
ViolenceSense ML Service - FastAPI Application

This is the main entry point for the ML inference service.
Provides REST API endpoints for model management and video inference.
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.config import settings
from app.models import model_manager
from app.inference import inference_pipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ViolenceSense ML Service",
    description="AI-powered video violence detection inference service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Startup Event ==============

@app.on_event("startup")
async def startup_event():
    """Auto-load model from environment configuration on startup."""
    model_path = settings.default_model_path
    architecture = settings.model_architecture
    
    logger.info(f"Startup - Checking for model at: {model_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Check models directory
    if os.path.exists("models"):
        logger.info(f"Models directory contents: {os.listdir('models')}")
    else:
        logger.warning("Models directory does not exist!")
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Auto-loading model from: {model_path}")
        logger.info(f"Architecture: {architecture}")
        
        try:
            result = model_manager.load_model(model_path, architecture)
            if result["success"]:
                logger.info("✓ Model auto-loaded successfully!")
            else:
                logger.warning(f"Model auto-load failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"Model auto-load failed: {e}")
    else:
        logger.warning(f"Model file not found at: {model_path}")
        logger.info("No default model configured or model file not found. Skipping auto-load.")


# ============== Pydantic Models ==============

class ModelLoadRequest(BaseModel):
    modelPath: str = Field(..., description="Path to the model file (.h5, .pth)")
    architecture: str = Field(default="keras-cnn", description="Model architecture")


class InferenceRequest(BaseModel):
    videoPath: str = Field(..., description="Path to the video file")
    modelPath: Optional[str] = Field(None, description="Optional model path")
    architecture: Optional[str] = Field(None, description="Model architecture")
    numFrames: Optional[int] = Field(None, description="Number of frames to process")


class HealthResponse(BaseModel):
    status: str
    message: str


class ModelStatusResponse(BaseModel):
    isLoaded: bool
    currentModel: Optional[dict]
    gpuAvailable: bool
    gpuMemory: Optional[dict]


# ============== Health Endpoints ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health status."""
    return {
        "status": "healthy",
        "message": "ViolenceSense ML Service is running"
    }


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ViolenceSense ML Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "model_load": "/model/load",
            "model_status": "/model/status",
            "model_metrics": "/model/metrics",
            "inference": "/inference/predict"
        }
    }


# ============== Model Management Endpoints ==============

@app.post("/model/load", tags=["Model"])
async def load_model(request: ModelLoadRequest):
    """
    Load a PyTorch model from the specified path.
    
    - **modelPath**: Absolute path to the .pth model file
    - **architecture**: Model architecture (videomae, timesformer, slowfast, resnet3d, i3d, custom)
    """
    try:
        result = model_manager.load_model(
            request.modelPath,
            request.architecture
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to load model"))
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status", response_model=ModelStatusResponse, tags=["Model"])
async def get_model_status():
    """Get current model status and GPU information."""
    return model_manager.get_status()


@app.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    """Get model performance metrics."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    return model_manager.get_metrics()


@app.post("/model/unload", tags=["Model"])
async def unload_model():
    """Unload the current model and free memory."""
    success = model_manager.unload_model()
    
    if success:
        return {"success": True, "message": "Model unloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to unload model")


# ============== Inference Endpoints ==============

@app.post("/inference/predict", tags=["Inference"])
async def predict(request: InferenceRequest):
    """
    Run violence detection inference on a video (local path).
    
    - **videoPath**: Path to the video file (mp4, avi, mov)
    - **modelPath**: Optional path to load a different model
    - **architecture**: Model architecture if loading new model
    - **numFrames**: Number of frames to process (default: 20)
    """
    try:
        # Validate video path
        if not os.path.exists(request.videoPath):
            raise HTTPException(status_code=400, detail=f"Video file not found: {request.videoPath}")
        
        # Run inference
        result = inference_pipeline.predict(
            video_path=request.videoPath,
            model_path=request.modelPath,
            architecture=request.architecture,
            num_frames=request.numFrames
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/predict-upload", tags=["Inference"])
async def predict_upload(
    video: UploadFile = File(..., description="Video file to analyze"),
    numFrames: Optional[int] = Form(None, description="Number of frames to process")
):
    """
    Run violence detection inference on an uploaded video file.
    Use this endpoint when sending video files from remote services.
    
    - **video**: Video file (mp4, avi, mov)
    - **numFrames**: Number of frames to process (default: 20)
    """
    temp_path = None
    try:
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = Path(video.filename).suffix.lower() if video.filename else '.mp4'
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(video.file, temp_file)
        
        logger.info(f"Received video upload: {video.filename}, saved to: {temp_path}")
        
        # Run inference
        result = inference_pipeline.predict(
            video_path=temp_path,
            num_frames=numFrames
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@app.post("/inference/batch", tags=["Inference"])
async def batch_predict(video_paths: List[str]):
    """
    Run inference on multiple videos.
    
    - **video_paths**: List of video file paths
    """
    try:
        # Validate paths
        for path in video_paths:
            if not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Video file not found: {path}")
        
        results = inference_pipeline.batch_predict(video_paths)
        return {"success": True, "results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Main Entry Point ==============

if __name__ == "__main__":
    logger.info(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🧠 ViolenceSense ML Service                             ║
║                                                           ║
║   Starting inference service...                           ║
║   Host: {settings.host}                                   ║
║   Port: {settings.port}                                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )
