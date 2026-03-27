"""
Adaptive Load Controller
========================
Dynamically adjusts streaming and AI parameters based on system load.

Key Functions:
1. Monitor CPU/GPU usage
2. Adjust stream quality when load is high
3. Reduce AI inference rate under stress
4. Ensure smooth playback even with 15 cameras
"""

import asyncio
import threading
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Callable, Any
from enum import Enum
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not installed - GPU monitoring disabled")


class LoadLevel(Enum):
    """System load levels."""
    LOW = "low"          # < 40% - Full quality
    MEDIUM = "medium"    # 40-60% - Slight reduction
    HIGH = "high"        # 60-80% - Significant reduction
    CRITICAL = "critical"  # > 80% - Minimal processing


@dataclass
class LoadMetrics:
    """Current system load metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    active_streams: int = 0
    active_viewers: int = 0
    inference_queue_size: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def overall_load(self) -> float:
        """Calculate overall system load (0-100)."""
        # Weight: CPU 40%, GPU 40%, Memory 20%
        if GPU_AVAILABLE and self.gpu_percent > 0:
            return (
                self.cpu_percent * 0.3 +
                self.gpu_percent * 0.4 +
                self.memory_percent * 0.2 +
                self.gpu_memory_percent * 0.1
            )
        else:
            return (
                self.cpu_percent * 0.6 +
                self.memory_percent * 0.4
            )
            
    @property
    def load_level(self) -> LoadLevel:
        """Determine load level."""
        load = self.overall_load
        if load < 40:
            return LoadLevel.LOW
        elif load < 60:
            return LoadLevel.MEDIUM
        elif load < 80:
            return LoadLevel.HIGH
        else:
            return LoadLevel.CRITICAL


@dataclass
class AdaptiveConfig:
    """Configuration adjusted by load controller."""
    # Display settings
    display_fps: int = 30
    display_quality: int = 85  # JPEG quality
    
    # AI settings
    ai_inference_interval_ms: int = 200
    ai_sample_rate: int = 5  # Frames per second for AI
    ai_batch_size: int = 1
    
    # Stream settings
    max_concurrent_streams: int = 15
    max_viewers_per_stream: int = 10
    
    # Quality level (1-5)
    quality_level: int = 5


# Predefined configs for each load level
LOAD_CONFIGS: Dict[LoadLevel, AdaptiveConfig] = {
    LoadLevel.LOW: AdaptiveConfig(
        display_fps=30,
        display_quality=85,
        ai_inference_interval_ms=200,
        ai_sample_rate=5,
        ai_batch_size=1,
        max_concurrent_streams=15,
        quality_level=5,
    ),
    LoadLevel.MEDIUM: AdaptiveConfig(
        display_fps=25,
        display_quality=75,
        ai_inference_interval_ms=300,
        ai_sample_rate=4,
        ai_batch_size=1,
        max_concurrent_streams=12,
        quality_level=4,
    ),
    LoadLevel.HIGH: AdaptiveConfig(
        display_fps=20,
        display_quality=65,
        ai_inference_interval_ms=400,
        ai_sample_rate=3,
        ai_batch_size=1,
        max_concurrent_streams=10,
        quality_level=3,
    ),
    LoadLevel.CRITICAL: AdaptiveConfig(
        display_fps=15,
        display_quality=50,
        ai_inference_interval_ms=500,
        ai_sample_rate=2,
        ai_batch_size=1,
        max_concurrent_streams=8,
        quality_level=2,
    ),
}


class LoadController:
    """
    Adaptive load controller for multi-camera streaming.
    
    Monitors system resources and adjusts streaming/AI parameters
    to maintain smooth performance even under high load.
    """
    
    def __init__(
        self,
        check_interval: float = 2.0,
        smoothing_window: int = 5,
        log_cooldown: float = 60.0  # Only log level changes every 60 seconds
    ):
        self.check_interval = check_interval
        self.smoothing_window = smoothing_window
        self.log_cooldown = log_cooldown
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Current state
        self.current_metrics = LoadMetrics()
        self.current_config = LOAD_CONFIGS[LoadLevel.LOW]
        self.current_level = LoadLevel.LOW
        
        # History for smoothing
        self._load_history: List[float] = []
        
        # Logging cooldown tracking
        self._last_log_time: float = 0.0
        self._startup_logged: bool = False
        
        # Callbacks
        self._on_config_change: Optional[Callable[[AdaptiveConfig], None]] = None
        self._on_level_change: Optional[Callable[[LoadLevel], None]] = None
        
        # External data sources
        self._stream_count_getter: Optional[Callable[[], int]] = None
        self._viewer_count_getter: Optional[Callable[[], int]] = None
        self._queue_size_getter: Optional[Callable[[], int]] = None
        
    async def start(self):
        """Start load monitoring."""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Load controller started")
        
    async def stop(self):
        """Stop load monitoring."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Load controller stopped")
        
    def set_config_change_callback(self, callback: Callable[[AdaptiveConfig], None]):
        """Set callback for config changes."""
        self._on_config_change = callback
        
    def set_level_change_callback(self, callback: Callable[[LoadLevel], None]):
        """Set callback for load level changes."""
        self._on_level_change = callback
        
    def set_stream_count_getter(self, getter: Callable[[], int]):
        """Set function to get active stream count."""
        self._stream_count_getter = getter
        
    def set_viewer_count_getter(self, getter: Callable[[], int]):
        """Set function to get total viewer count."""
        self._viewer_count_getter = getter
        
    def set_queue_size_getter(self, getter: Callable[[], int]):
        """Set function to get inference queue size."""
        self._queue_size_getter = getter
        
    def _collect_metrics(self) -> LoadMetrics:
        """Collect current system metrics."""
        metrics = LoadMetrics()
        
        # CPU and memory
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.memory_percent = psutil.virtual_memory().percent
        
        # GPU (if available)
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics.gpu_percent = gpu.load * 100
                    metrics.gpu_memory_percent = gpu.memoryUtil * 100
            except Exception as e:
                logger.debug(f"GPU metrics error: {e}")
                
        # Application metrics
        if self._stream_count_getter:
            metrics.active_streams = self._stream_count_getter()
        if self._viewer_count_getter:
            metrics.active_viewers = self._viewer_count_getter()
        if self._queue_size_getter:
            metrics.inference_queue_size = self._queue_size_getter()
            
        metrics.timestamp = datetime.utcnow()
        return metrics
        
    def _get_smoothed_load(self, new_load: float) -> float:
        """Get smoothed load value to avoid oscillation."""
        self._load_history.append(new_load)
        if len(self._load_history) > self.smoothing_window:
            self._load_history = self._load_history[-self.smoothing_window:]
        return sum(self._load_history) / len(self._load_history)
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Smooth the load value
                smoothed_load = self._get_smoothed_load(metrics.overall_load)
                
                # Determine load level based on smoothed value
                if smoothed_load < 40:
                    new_level = LoadLevel.LOW
                elif smoothed_load < 60:
                    new_level = LoadLevel.MEDIUM
                elif smoothed_load < 80:
                    new_level = LoadLevel.HIGH
                else:
                    new_level = LoadLevel.CRITICAL
                
                # Log initial load level only once at startup
                current_time = time.time()
                if not self._startup_logged:
                    logger.info(f"Initial load level: {new_level.value} (load: {smoothed_load:.1f}%)")
                    self._startup_logged = True
                    self._last_log_time = current_time
                    
                # Update state
                with self._lock:
                    self.current_metrics = metrics
                    
                    # Only change level if it's different (with hysteresis)
                    if new_level != self.current_level:
                        # Apply hysteresis - need sustained change
                        level_order = [LoadLevel.LOW, LoadLevel.MEDIUM, LoadLevel.HIGH, LoadLevel.CRITICAL]
                        current_idx = level_order.index(self.current_level)
                        new_idx = level_order.index(new_level)
                        
                        # Check if we should log this change (cooldown check)
                        should_log = (current_time - self._last_log_time) >= self.log_cooldown
                        
                        # Require larger change to increase level (aggressive)
                        # Require smaller change to decrease level (conservative)
                        if new_idx > current_idx:  # Load increasing
                            # Apply change immediately
                            self.current_level = new_level
                            self.current_config = LOAD_CONFIGS[new_level]
                            
                            if self._on_level_change:
                                asyncio.create_task(
                                    asyncio.to_thread(self._on_level_change, new_level)
                                )
                            if self._on_config_change:
                                asyncio.create_task(
                                    asyncio.to_thread(self._on_config_change, self.current_config)
                                )
                            
                            # Only log if cooldown has passed (avoid spam)
                            if should_log:
                                logger.warning(f"Load level increased to {new_level.value} (load: {smoothed_load:.1f}%)")
                                self._last_log_time = current_time
                            
                        elif new_idx < current_idx:  # Load decreasing
                            # Be more conservative when decreasing
                            # Only decrease one level at a time
                            self.current_level = level_order[current_idx - 1]
                            self.current_config = LOAD_CONFIGS[self.current_level]
                            
                            if self._on_level_change:
                                asyncio.create_task(
                                    asyncio.to_thread(self._on_level_change, self.current_level)
                                )
                            if self._on_config_change:
                                asyncio.create_task(
                                    asyncio.to_thread(self._on_config_change, self.current_config)
                                )
                            
                            # Only log if cooldown has passed (avoid spam)
                            if should_log:
                                logger.info(f"Load level decreased to {self.current_level.value} (load: {smoothed_load:.1f}%)")
                                self._last_log_time = current_time
                            
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Load monitor error: {e}")
                await asyncio.sleep(self.check_interval)
                
    def get_current_config(self) -> AdaptiveConfig:
        """Get current adaptive configuration."""
        with self._lock:
            return self.current_config
            
    def get_current_metrics(self) -> LoadMetrics:
        """Get current load metrics."""
        with self._lock:
            return self.current_metrics
            
    def get_status(self) -> Dict[str, Any]:
        """Get load controller status."""
        with self._lock:
            return {
                "load_level": self.current_level.value,
                "overall_load": round(self.current_metrics.overall_load, 1),
                "cpu_percent": round(self.current_metrics.cpu_percent, 1),
                "memory_percent": round(self.current_metrics.memory_percent, 1),
                "gpu_percent": round(self.current_metrics.gpu_percent, 1) if GPU_AVAILABLE else None,
                "active_streams": self.current_metrics.active_streams,
                "active_viewers": self.current_metrics.active_viewers,
                "config": {
                    "display_fps": self.current_config.display_fps,
                    "display_quality": self.current_config.display_quality,
                    "ai_interval_ms": self.current_config.ai_inference_interval_ms,
                    "ai_sample_rate": self.current_config.ai_sample_rate,
                    "quality_level": self.current_config.quality_level,
                },
            }
            
    def force_level(self, level: LoadLevel):
        """Force a specific load level (for testing/override)."""
        with self._lock:
            self.current_level = level
            self.current_config = LOAD_CONFIGS[level]
            logger.info(f"Load level forced to {level.value}")


# Global instance
_load_controller: Optional[LoadController] = None


def get_load_controller() -> LoadController:
    """Get or create the global load controller."""
    global _load_controller
    if _load_controller is None:
        _load_controller = LoadController()
    return _load_controller


async def init_load_controller():
    """Initialize and start the load controller."""
    controller = get_load_controller()
    await controller.start()
    return controller
