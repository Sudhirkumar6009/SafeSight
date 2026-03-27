"""
Ultra Low Latency Streaming API Routes
======================================
WebRTC and streaming endpoints for SafeSight.

Architecture:
    Cameras (RTSP) --> MediaMTX --> WebRTC --> Frontend (200-500ms latency)
                                --> Frame Sampler --> AI Workers
"""

import asyncio
import aiohttp
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.streaming.webrtc_service import (
    get_webrtc_manager, 
    WebRTCStreamManager,
    StreamQuality
)
from app.streaming.frame_sampler import get_sampler_pool, SamplerConfig
from app.streaming.load_controller import get_load_controller, LoadLevel

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])

# MediaMTX settings
MEDIAMTX_HOST = os.getenv("MEDIAMTX_HOST", "localhost")
MEDIAMTX_WEBRTC_PORT = int(os.getenv("MEDIAMTX_WEBRTC_PORT", "8889"))


# ============== Models ==============

class WebRTCStreamCreate(BaseModel):
    """Request to create a WebRTC stream."""
    stream_id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Display name")
    rtsp_url: str = Field(..., description="Camera RTSP URL")
    quality: str = Field(default="high", description="Quality: ultra, high, medium, low, minimal")
    enable_ai: bool = Field(default=True, description="Enable AI processing")


class WebRTCOfferRequest(BaseModel):
    """WebRTC SDP offer from browser."""
    sdp: str
    type: str = "offer"


# ============== WebRTC Stream Management ==============

@router.post("/webrtc/streams")
async def create_webrtc_stream(request: WebRTCStreamCreate):
    """
    Add a camera stream for WebRTC delivery.
    
    Creates:
    1. MediaMTX path for RTSP-to-WebRTC conversion
    2. Frame sampler for AI pipeline (if enabled)
    
    Returns URLs for WebRTC, RTSP proxy, and HLS fallback.
    """
    manager = get_webrtc_manager()
    
    # Parse quality
    try:
        quality = StreamQuality(request.quality)
    except ValueError:
        quality = StreamQuality.HIGH
        
    # Add to MediaMTX
    urls = await manager.add_stream(
        stream_id=request.stream_id,
        name=request.name,
        rtsp_url=request.rtsp_url,
        quality=quality
    )
    
    # Add AI frame sampler if enabled
    if request.enable_ai:
        pool = get_sampler_pool()
        rtsp_proxy = urls.get("rtsp_proxy_url", request.rtsp_url)
        pool.add_sampler(
            stream_id=request.stream_id,
            rtsp_url=rtsp_proxy,
            target_fps=5.0,  # AI sampling rate
            auto_start=True
        )
        
    return {
        "success": True,
        "stream_id": request.stream_id,
        "urls": urls,
        "ai_enabled": request.enable_ai,
    }


@router.delete("/webrtc/streams/{stream_id}")
async def remove_webrtc_stream(stream_id: str):
    """Remove a WebRTC stream."""
    manager = get_webrtc_manager()
    pool = get_sampler_pool()
    
    # Remove from both managers
    await manager.remove_stream(stream_id)
    pool.remove_sampler(stream_id)
    
    return {"success": True, "message": f"Stream {stream_id} removed"}


@router.get("/webrtc/streams")
async def list_webrtc_streams():
    """List all WebRTC streams with status."""
    manager = get_webrtc_manager()
    streams = await manager.list_all_streams()
    
    return {
        "success": True,
        "data": streams,
        "count": len(streams)
    }


@router.get("/webrtc/streams/{stream_id}")
async def get_webrtc_stream(stream_id: str):
    """Get WebRTC stream status and URLs."""
    manager = get_webrtc_manager()
    status = await manager.get_stream_status(stream_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Stream not found")
        
    return {"success": True, "data": status}


@router.get("/webrtc/streams/{stream_id}/urls")
async def get_stream_urls(stream_id: str):
    """Get all playback URLs for a stream."""
    manager = get_webrtc_manager()
    urls = manager.get_stream_urls(stream_id)
    
    if not urls:
        raise HTTPException(status_code=404, detail="Stream not found")
        
    return {"success": True, "data": urls}


# ============== WHEP Signaling Proxy ==============

@router.post("/webrtc/streams/{stream_id}/whep")
async def webrtc_whep_offer(stream_id: str, request: Request):
    """
    WebRTC WHEP signaling endpoint.
    
    Proxies WebRTC offer/answer to MediaMTX for ultra-low latency streaming.
    This is the PRIMARY method for browser playback.
    
    Protocol: WebRTC-HTTP Egress Protocol (WHEP)
    Latency: ~200-500ms
    """
    # Get SDP offer from request body
    content_type = request.headers.get("content-type", "")
    
    if "application/sdp" in content_type:
        sdp_offer = await request.body()
    else:
        body = await request.json()
        sdp_offer = body.get("sdp", "").encode()
        
    if not sdp_offer:
        raise HTTPException(status_code=400, detail="Missing SDP offer")
        
    # Build MediaMTX WHEP URL
    path_name = f"cam_{stream_id}"
    whep_url = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_WEBRTC_PORT}/{path_name}/whep"
    
    # Proxy to MediaMTX
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                whep_url,
                data=sdp_offer,
                headers={
                    "Content-Type": "application/sdp",
                    "Accept": "application/sdp",
                }
            ) as resp:
                if resp.status == 201:
                    sdp_answer = await resp.read()
                    
                    # Extract resource URL for ICE candidates
                    location = resp.headers.get("Location", "")
                    
                    return Response(
                        content=sdp_answer,
                        status_code=201,
                        media_type="application/sdp",
                        headers={
                            "Location": location,
                            "Access-Control-Expose-Headers": "Location",
                        }
                    )
                else:
                    error = await resp.text()
                    logger.error(f"WHEP error: {resp.status} - {error}")
                    raise HTTPException(
                        status_code=resp.status,
                        detail=f"MediaMTX WHEP error: {error}"
                    )
                    
    except aiohttp.ClientError as e:
        logger.error(f"WHEP connection error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to MediaMTX: {e}"
        )


@router.patch("/webrtc/streams/{stream_id}/whep/{resource_id}")
async def webrtc_whep_patch(stream_id: str, resource_id: str, request: Request):
    """
    WHEP ICE candidate trickle endpoint.
    
    Forwards ICE candidates to MediaMTX for NAT traversal.
    """
    body = await request.body()
    
    path_name = f"cam_{stream_id}"
    patch_url = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_WEBRTC_PORT}/{path_name}/whep/{resource_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                patch_url,
                data=body,
                headers={"Content-Type": request.headers.get("content-type", "application/trickle-ice-sdpfrag")}
            ) as resp:
                response_body = await resp.read()
                return Response(
                    content=response_body,
                    status_code=resp.status,
                    media_type=resp.content_type
                )
    except Exception as e:
        logger.error(f"WHEP patch error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/webrtc/streams/{stream_id}/whep/{resource_id}")
async def webrtc_whep_delete(stream_id: str, resource_id: str):
    """
    WHEP session teardown.
    
    Closes WebRTC connection when viewer leaves.
    """
    path_name = f"cam_{stream_id}"
    delete_url = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_WEBRTC_PORT}/{path_name}/whep/{resource_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(delete_url) as resp:
                return Response(status_code=resp.status)
    except Exception as e:
        logger.error(f"WHEP delete error: {e}")
        return Response(status_code=200)


# ============== Direct MediaMTX Proxy ==============

@router.api_route(
    "/mediamtx/{path:path}",
    methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"]
)
async def proxy_mediamtx(path: str, request: Request):
    """
    Proxy all requests to MediaMTX WebRTC server.
    
    This allows the frontend to connect directly to MediaMTX
    for WebRTC signaling without CORS issues.
    """
    # Build target URL
    target_url = f"http://{MEDIAMTX_HOST}:{MEDIAMTX_WEBRTC_PORT}/{path}"
    
    # Get request body
    body = await request.body() if request.method in ["POST", "PATCH"] else None
    
    # Forward headers (except host)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host"]}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=target_url,
                data=body,
                headers=headers,
                allow_redirects=False
            ) as resp:
                response_body = await resp.read()
                
                # Forward response headers
                response_headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ["transfer-encoding", "content-encoding"]
                }
                
                # Add CORS headers
                response_headers["Access-Control-Allow-Origin"] = "*"
                response_headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"
                response_headers["Access-Control-Allow-Headers"] = "Content-Type"
                response_headers["Access-Control-Expose-Headers"] = "Location"
                
                return Response(
                    content=response_body,
                    status_code=resp.status,
                    headers=response_headers,
                    media_type=resp.content_type
                )
                
    except aiohttp.ClientError as e:
        logger.error(f"MediaMTX proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"MediaMTX unavailable: {e}")


# ============== Load Control ==============

@router.get("/load/status")
async def get_load_status():
    """Get current system load and adaptive config."""
    controller = get_load_controller()
    return {
        "success": True,
        "data": controller.get_status()
    }


@router.post("/load/level/{level}")
async def set_load_level(level: str):
    """Force a specific load level (for testing)."""
    try:
        load_level = LoadLevel(level)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid level: {level}")
        
    controller = get_load_controller()
    controller.force_level(load_level)
    
    return {
        "success": True,
        "message": f"Load level set to {level}",
        "config": controller.get_status()
    }


@router.post("/quality/{stream_id}/{quality}")
async def set_stream_quality(stream_id: str, quality: str):
    """Set quality for a specific stream."""
    try:
        quality_level = StreamQuality(quality)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid quality: {quality}")
        
    manager = get_webrtc_manager()
    await manager.set_stream_quality(stream_id, quality_level)
    
    return {
        "success": True,
        "message": f"Stream {stream_id} quality set to {quality}"
    }


# ============== AI Frame Samplers ==============

@router.get("/samplers")
async def list_frame_samplers():
    """List all AI frame samplers."""
    pool = get_sampler_pool()
    return {
        "success": True,
        "data": pool.get_all_status()
    }


@router.get("/samplers/{stream_id}")
async def get_sampler_status(stream_id: str):
    """Get status of a specific frame sampler."""
    pool = get_sampler_pool()
    sampler = pool.get_sampler(stream_id)
    
    if not sampler:
        raise HTTPException(status_code=404, detail="Sampler not found")
        
    return {
        "success": True,
        "data": sampler.get_status()
    }


@router.post("/samplers/{stream_id}/quality/{level}")
async def set_sampler_quality(stream_id: str, level: int):
    """Set quality level for a frame sampler (1-5)."""
    pool = get_sampler_pool()
    sampler = pool.get_sampler(stream_id)
    
    if not sampler:
        raise HTTPException(status_code=404, detail="Sampler not found")
        
    sampler.set_quality_level(level)
    
    return {
        "success": True,
        "message": f"Sampler {stream_id} quality set to level {level}"
    }


# ============== Initialization ==============

async def init_streaming_services():
    """Initialize all streaming services."""
    # Start WebRTC manager
    manager = get_webrtc_manager()
    await manager.start()
    
    # Start load controller
    controller = get_load_controller()
    await controller.start()
    
    # Wire up callbacks
    controller.set_stream_count_getter(lambda: len(manager.streams))
    
    logger.info("Streaming services initialized")


async def shutdown_streaming_services():
    """Shutdown all streaming services."""
    manager = get_webrtc_manager()
    await manager.stop()
    
    controller = get_load_controller()
    await controller.stop()
    
    pool = get_sampler_pool()
    pool.stop_all()
    
    logger.info("Streaming services shutdown")
