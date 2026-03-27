"""
SafeSight ML Service - FastAPI Application
==========================================
ONNX-based violence detection inference service.

Clean implementation based on reference code:
- ONNX Runtime for fast, portable inference
- 16-frame input, single sigmoid output
- Simple preprocessing: resize 224x224, BGR->RGB, normalize [0,1]
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
from app.inference import inference_pipeline, get_detector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="SafeSight ML Service",
    description="ONNX-powered violence detection inference service",
    version="2.0.0",
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
    """Auto-load ONNX model on startup."""
    model_path = settings.default_model_path
    
    logger.info(f"""
╔═══════════════════════════════════════════════════════════╗
║   SafeSight ML Service - ONNX Violence Detection          ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check models directory
    if os.path.exists("models"):
        logger.info(f"Models directory: {os.listdir('models')}")
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading ONNX model: {model_path}")
        
        detector = get_detector()
        result = detector.load_model(model_path)
        
        if result["success"]:
            logger.info("Model loaded successfully!")
            logger.info(f"Providers: {detector.providers_used}")
        else:
            logger.warning(f"Model load failed: {result.get('error')}")
    else:
        logger.warning(f"Model not found: {model_path}")


# ============== Pydantic Models ==============

class ModelLoadRequest(BaseModel):
    modelPath: str = Field(..., description="Path to the ONNX model file")


class InferenceRequest(BaseModel):
    videoPath: str = Field(..., description="Path to the video file")


class HealthResponse(BaseModel):
    status: str
    message: str


class ModelStatusResponse(BaseModel):
    isLoaded: bool
    modelPath: Optional[str]
    providers: List[str]
    config: dict


# ============== Health Endpoints ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health status."""
    return {
        "status": "healthy",
        "message": "SafeSight ML Service is running"
    }


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with service info."""
    detector = get_detector()
    return {
        "service": "SafeSight ML Service",
        "version": "2.0.0",
        "engine": "ONNX Runtime",
        "model_loaded": detector.is_loaded,
        "providers": detector.providers_used,
        "endpoints": {
            "health": "/health",
            "model_load": "/model/load",
            "model_status": "/model/status",
            "inference": "/inference/predict",
            "inference_upload": "/inference/predict-upload"
        }
    }


# ============== Model Management Endpoints ==============

@app.post("/model/load", tags=["Model"])
async def load_model(request: ModelLoadRequest):
    """
    Load an ONNX model from the specified path.
    
    - **modelPath**: Path to the .onnx model file
    """
    try:
        detector = get_detector()
        result = detector.load_model(request.modelPath)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status", tags=["Model"])
async def get_model_status():
    """Get current model status."""
    detector = get_detector()
    status = detector.get_status()
    
    return {
        "isLoaded": status["is_loaded"],
        "modelPath": status["model_path"],
        "providers": status["providers"],
        "config": status["config"]
    }


@app.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    """Get model performance metrics."""
    detector = get_detector()
    
    if not detector.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    return {
        "is_loaded": detector.is_loaded,
        "providers": detector.providers_used,
        "config": {
            "seq_len": detector.SEQ_LEN,
            "img_size": detector.IMG_SIZE,
            "frame_skip": detector.FRAME_SKIP
        }
    }


@app.post("/model/unload", tags=["Model"])
async def unload_model():
    """Unload the current model and free memory."""
    detector = get_detector()
    detector.unload_model()
    
    return {"success": True, "message": "Model unloaded successfully"}


# ============== Inference Endpoints ==============

@app.post("/inference/predict", tags=["Inference"])
async def predict(request: InferenceRequest):
    """
    Run violence detection on a video file (local path).
    
    - **videoPath**: Path to the video file (mp4, avi, mov)
    """
    try:
        # Validate video path
        if not os.path.exists(request.videoPath):
            raise HTTPException(
                status_code=400, 
                detail=f"Video file not found: {request.videoPath}"
            )
        
        # Run inference
        result = inference_pipeline.predict(video_path=request.videoPath)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Inference failed")
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/predict-upload", tags=["Inference"])
async def predict_upload(
    video: UploadFile = File(..., description="Video file to analyze")
):
    """
    Run violence detection on an uploaded video file.
    
    - **video**: Video file (mp4, avi, mov, mkv, webm)
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
        
        logger.info(f"Received video: {video.filename}, saved to: {temp_path}")
        
        # Run inference
        result = inference_pipeline.predict(video_path=temp_path)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Inference failed")
            )
        
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
                raise HTTPException(
                    status_code=400, 
                    detail=f"Video file not found: {path}"
                )
        
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
║   SafeSight ML Service - ONNX Violence Detection          ║
║                                                           ║
║   Host: {settings.host}                                   ║
║   Port: {settings.port}                                   ║
║   Model: {settings.default_model_path}                    ║
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
