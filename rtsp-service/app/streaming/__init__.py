"""
SafeSight Streaming Module
==========================
Ultra-low latency streaming architecture using MediaMTX + WebRTC.

Architecture:
    Cameras (RTSP) --> MediaMTX --> WebRTC --> Browser (~200-500ms latency)
                                --> Frame Sampler --> AI Worker Pool

Components:
1. WebRTC Service - Manages MediaMTX integration for ultra-low latency
2. Frame Sampler - Decoupled AI frame extraction from display
3. Load Controller - Adaptive quality based on system load
4. MediaMTX Manager - Direct MediaMTX API communication
"""

from .mediamtx_manager import MediaMTXManager, get_mediamtx_manager
from .webrtc_service import (
    WebRTCStreamManager,
    get_webrtc_manager,
    init_webrtc_manager,
    StreamQuality,
    QualityConfig,
    WebRTCStream,
)
from .frame_sampler import (
    FrameSampler,
    FrameSamplerPool,
    get_sampler_pool,
    SampledFrame,
    SamplerConfig,
)
from .load_controller import (
    LoadController,
    get_load_controller,
    init_load_controller,
    LoadLevel,
    LoadMetrics,
    AdaptiveConfig,
)
from .routes import (
    router as streaming_router,
    init_streaming_services,
    shutdown_streaming_services,
)

__all__ = [
    # MediaMTX Manager
    "MediaMTXManager",
    "get_mediamtx_manager",
    # WebRTC Service
    "WebRTCStreamManager",
    "get_webrtc_manager",
    "init_webrtc_manager",
    "StreamQuality",
    "QualityConfig",
    "WebRTCStream",
    # Frame Sampler
    "FrameSampler",
    "FrameSamplerPool",
    "get_sampler_pool",
    "SampledFrame",
    "SamplerConfig",
    # Load Controller
    "LoadController",
    "get_load_controller",
    "init_load_controller",
    "LoadLevel",
    "LoadMetrics",
    "AdaptiveConfig",
    # Routes
    "streaming_router",
    "init_streaming_services",
    "shutdown_streaming_services",
]
