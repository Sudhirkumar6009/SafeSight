"""
ViolenceSense - Event Detection Engine
======================================
Implements threshold + duration rules to convert raw inference scores
into meaningful violence events.

Key Features:
- Configurable threshold and duration rules
- Hysteresis to prevent event flickering
- Cooldown periods between events
- Automatic clip recording on event trigger
- Severity calculation based on confidence
- Real-time WebSocket notifications
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

import numpy as np
from loguru import logger

from app.config import settings
from app.db import (
    EventRepository, InferenceLogRepository, StreamRepository,
    EventStatus, EventSeverity
)
from app.stream.ffmpeg_ingestion import FFmpegIngestion, FramePacket, ClipRecorder

# Simple per-stream debounce state
_debounce_state = {}  # {stream_id: {"high_count": int}}

MIN_CONSECUTIVE_FRAMES = 8
MAX_MOTION_SCORE = 0.25  # ignore if scene motion above this

def _process_frame_for_event(stream_id, frame, violence_score, detect_fn, *args, **kwargs):
    state = _debounce_state.setdefault(stream_id, {"high_count": 0})
    # TODO: Implement motion analysis or import from correct module
    motion = 0.0  # Placeholder: implement motion detection

    # If large scene motion (camera move/scene change), reset and skip
    if motion is None or motion > MAX_MOTION_SCORE:
        state["high_count"] = 0
        return None

    if violence_score >= detect_fn.threshold:  # or use configured threshold
        state["high_count"] += 1
    else:
        state["high_count"] = 0

    # Require sustained high scores before triggering
    if state["high_count"] >= MIN_CONSECUTIVE_FRAMES:
        state["high_count"] = 0  # reset after trigger
        return detect_fn.trigger_event(stream_id, frame, violence_score, *args, **kwargs)

    return None


@dataclass
class InferenceScore:
    """Single inference result from the ML pipeline."""
    violence_score: float
    non_violence_score: float
    timestamp: datetime
    inference_time_ms: float
    frame_count: int
    window_start: datetime
    window_end: datetime
    
    @property
    def is_violent(self) -> bool:
        """Check if score exceeds threshold."""
        return self.violence_score >= settings.violence_threshold
    
    @property
    def classification(self) -> str:
        return "violence" if self.is_violent else "non-violence"


@dataclass
class DetectorState:
    """
    Tracks the current state of event detection for a stream.
    Uses a state machine approach for clean event lifecycle management.
    """
    class Phase(str, Enum):
        IDLE = "idle"           # Waiting for violence detection
        TRIGGERED = "triggered"  # Above threshold, counting consecutive
        ACTIVE = "active"       # Event confirmed and recording
        ENDING = "ending"       # Below threshold, waiting to end
        COOLDOWN = "cooldown"   # Event ended, in cooldown period
    
    phase: Phase = Phase.IDLE
    
    # Trigger tracking
    consecutive_violent_count: int = 0
    last_violent_time: Optional[datetime] = None
    
    # Active event
    current_event_id: Optional[str] = None
    event_start_time: Optional[datetime] = None
    event_scores: List[float] = field(default_factory=list)
    event_frame_count: int = 0
    peak_score: float = 0.0
    
    # Cooldown tracking
    cooldown_until: Optional[datetime] = None
    last_event_end: Optional[datetime] = None
    
    # Frame buffer for clips (separate from inference buffer)
    clip_frames: List[FramePacket] = field(default_factory=list)
    
    def reset_trigger(self):
        """Reset trigger phase counters."""
        self.consecutive_violent_count = 0
        self.last_violent_time = None
    
    def reset_event(self):
        """Reset event state after event ends."""
        self.current_event_id = None
        self.event_start_time = None
        self.event_scores = []
        self.event_frame_count = 0
        self.peak_score = 0.0
        self.clip_frames = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for debugging."""
        return {
            "phase": self.phase.value,
            "consecutive_violent_count": self.consecutive_violent_count,
            "current_event_id": self.current_event_id,
            "event_scores_count": len(self.event_scores),
            "peak_score": self.peak_score,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None
        }


class EventDetectionEngine:
    """
    Event detection engine that converts raw inference scores into events.
    
    Detection Rules:
    1. TRIGGER: violence_score >= threshold
    2. CONFIRM: consecutive_triggers >= min_consecutive_frames
    3. END: violence_score < threshold * hysteresis_factor for duration
    4. COOLDOWN: no new events for cooldown_seconds after event ends
    
    Severity Mapping:
    - 0.65-0.75: LOW
    - 0.75-0.85: MEDIUM
    - 0.85-0.95: HIGH
    - 0.95+: CRITICAL
    """
    
    def __init__(
        self,
        stream_id: str,
        stream_name: str,
        ingestion: FFmpegIngestion,
        # Callbacks
        on_event_start: Optional[Callable[[str, Dict], None]] = None,
        on_event_end: Optional[Callable[[str, Dict], None]] = None,
        on_alert: Optional[Callable[[Dict], None]] = None,
        # Configuration overrides
        threshold: Optional[float] = None,
        min_duration_seconds: Optional[float] = None,
        cooldown_seconds: Optional[int] = None
    ):
        self.stream_id = stream_id
        self.stream_name = stream_name
        self.ingestion = ingestion
        
        # Callbacks
        self.on_event_start = on_event_start
        self.on_event_end = on_event_end
        self.on_alert = on_alert
        
        # Configuration
        self.threshold = threshold or settings.violence_threshold
        self.min_consecutive = settings.min_consecutive_frames
        self.min_duration_seconds = min_duration_seconds or 2.0
        self.cooldown_seconds = cooldown_seconds or settings.alert_cooldown_seconds
        self.hysteresis_factor = 0.8  # End event when score < threshold * this
        self.clip_before_seconds = settings.clip_duration_before
        self.clip_after_seconds = settings.clip_duration_after
        
        # State
        self.state = DetectorState()
        
        # Clip recorder
        self.clip_recorder = ClipRecorder(settings.clips_dir)
        
        # Pending tasks
        self._ending_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_inferences = 0
        self.total_events = 0
    
    async def process_score(self, score: InferenceScore) -> Optional[Dict]:
        """
        Process an inference score and manage event lifecycle.
        
        Returns event info dict if a new event started or ended.
        """
        self.total_inferences += 1
        now = score.timestamp
        
        # Log inference to database (async, fire and forget)
        asyncio.create_task(self._log_inference(score))
        
        # Check if in cooldown
        if self.state.phase == DetectorState.Phase.COOLDOWN:
            if self.state.cooldown_until and now >= self.state.cooldown_until:
                self.state.phase = DetectorState.Phase.IDLE
                logger.debug(f"Stream {self.stream_name}: Cooldown ended")
            else:
                return None  # Still in cooldown
        
        # Get pre-event frames for potential clip
        if self.state.phase in [DetectorState.Phase.IDLE, DetectorState.Phase.TRIGGERED]:
            pre_frames = self.ingestion.get_frame_window(self.clip_before_seconds)
            if pre_frames:
                self.state.clip_frames = pre_frames.copy()
        
        # Check score against threshold
        is_violent = score.violence_score >= self.threshold
        end_threshold = self.threshold * self.hysteresis_factor
        is_below_end = score.violence_score < end_threshold
        
        result = None
        
        # State machine transitions
        if self.state.phase == DetectorState.Phase.IDLE:
            if is_violent:
                result = await self._transition_to_triggered(score)
        
        elif self.state.phase == DetectorState.Phase.TRIGGERED:
            if is_violent:
                self.state.consecutive_violent_count += 1
                self.state.last_violent_time = now
                
                # Check if we should confirm the event
                if self.state.consecutive_violent_count >= self.min_consecutive:
                    result = await self._transition_to_active(score)
            else:
                # Reset if score drops before confirmation
                self.state.reset_trigger()
                self.state.phase = DetectorState.Phase.IDLE
        
        elif self.state.phase == DetectorState.Phase.ACTIVE:
            # Add score to event
            self.state.event_scores.append(score.violence_score)
            self.state.event_frame_count += score.frame_count
            self.state.peak_score = max(self.state.peak_score, score.violence_score)
            
            # Collect frames for clip
            latest = self.ingestion.ring_buffer.get_latest(1)
            if latest:
                self.state.clip_frames.extend(latest)
            
            if is_below_end:
                # Start ending process
                await self._transition_to_ending(score)
            else:
                # Reset ending timer if score goes back up
                if self._ending_task:
                    self._ending_task.cancel()
                    self._ending_task = None
        
        elif self.state.phase == DetectorState.Phase.ENDING:
            if is_violent:
                # Event resumed
                self.state.phase = DetectorState.Phase.ACTIVE
                if self._ending_task:
                    self._ending_task.cancel()
                    self._ending_task = None
                self.state.event_scores.append(score.violence_score)
            # If still below threshold, the ending task will complete
        
        return result
    
    async def _transition_to_triggered(self, score: InferenceScore) -> None:
        """Transition from IDLE to TRIGGERED."""
        self.state.phase = DetectorState.Phase.TRIGGERED
        self.state.consecutive_violent_count = 1
        self.state.last_violent_time = score.timestamp
        logger.debug(f"Stream {self.stream_name}: Violence detected, score={score.violence_score:.2%}")
        return None
    
    async def _transition_to_active(self, score: InferenceScore) -> Dict:
        """Transition from TRIGGERED to ACTIVE - create event."""
        self.state.phase = DetectorState.Phase.ACTIVE
        self.state.event_start_time = score.timestamp
        self.state.event_scores = [score.violence_score]
        self.state.peak_score = score.violence_score
        self.state.event_frame_count = score.frame_count
        
        # Create event in database
        try:
            event = await EventRepository.create(
                stream_id=self.stream_id,
                stream_name=self.stream_name,
                start_time=score.timestamp,
                max_confidence=score.violence_score,
                avg_confidence=score.violence_score,
                min_confidence=score.violence_score,
                frame_count=score.frame_count
            )
            self.state.current_event_id = str(event.id)
            self.total_events += 1
            
            logger.warning(
                f"🚨 VIOLENCE EVENT STARTED on {self.stream_name} "
                f"(ID: {self.state.current_event_id}, confidence: {score.violence_score:.2%})"
            )
            
            event_info = {
                "event_id": self.state.current_event_id,
                "stream_id": self.stream_id,
                "stream_name": self.stream_name,
                "start_time": score.timestamp.isoformat(),
                "confidence": score.violence_score,
                "severity": EventRepository.calculate_severity(score.violence_score)
            }
            
            # Notify callback
            if self.on_event_start:
                try:
                    self.on_event_start(self.stream_id, event_info)
                except Exception as e:
                    logger.error(f"Event start callback error: {e}")
            
            return event_info
            
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return None
    
    async def _transition_to_ending(self, score: InferenceScore) -> None:
        """Transition from ACTIVE to ENDING - schedule end."""
        self.state.phase = DetectorState.Phase.ENDING
        
        # Schedule event end after capturing more footage
        self._ending_task = asyncio.create_task(
            self._complete_event_ending(score)
        )
    
    async def _complete_event_ending(self, final_score: InferenceScore) -> None:
        """Complete event ending after delay to capture post-event footage."""
        try:
            # Wait to capture post-event footage
            await asyncio.sleep(self.clip_after_seconds)
            
            # If still in ENDING phase (not resumed), finalize
            if self.state.phase == DetectorState.Phase.ENDING:
                await self._finalize_event(final_score)
        except asyncio.CancelledError:
            # Event resumed, don't end
            logger.debug(f"Stream {self.stream_name}: Event ending cancelled (resumed)")
    
    async def _finalize_event(self, final_score: InferenceScore) -> None:
        """Finalize and close the current event."""
        if not self.state.current_event_id:
            return
        
        end_time = datetime.utcnow()
        scores = self.state.event_scores
        
        # Calculate statistics
        duration = (end_time - self.state.event_start_time).total_seconds() if self.state.event_start_time else 0
        
        logger.info(
            f"✅ VIOLENCE EVENT ENDED on {self.stream_name} "
            f"(duration: {duration:.1f}s, peak: {self.state.peak_score:.2%}, "
            f"avg: {sum(scores)/len(scores):.2%})"
        )
        
        # Save clip
        clip_path = None
        thumbnail_path = None
        clip_duration = None
        person_image_filenames = []
        person_count = 0
        
        if self.state.clip_frames:
            # Get additional frames after event
            post_frames = self.ingestion.get_frame_window(self.clip_after_seconds)
            all_frames = self.state.clip_frames + (post_frames or [])
            
            # Save clip
            clip_path = self.clip_recorder.save_clip(
                all_frames,
                self.stream_id,
                self.state.current_event_id
            )
            
            if clip_path:
                clip_duration = len(all_frames) // self.ingestion.config.target_fps
            
            # Save thumbnail from peak moment (middle of event)
            if all_frames:
                peak_idx = len(all_frames) // 2
                thumbnail_path = self.clip_recorder.save_thumbnail(
                    all_frames[peak_idx].frame,
                    self.stream_id,
                    self.state.current_event_id
                )
            
            # Capture person images from event frames
            try:
                from app.detection.person_capture import person_capture_engine
                raw_frames = [pkt.frame for pkt in all_frames if pkt.frame is not None]
                if raw_frames:
                    captures = person_capture_engine.capture_persons_from_frames(
                        frames=raw_frames,
                        event_id=self.state.current_event_id,
                        stream_id=self.stream_id
                    )
                    person_image_filenames = [Path(c.image_path).name for c in captures]
                    person_count = len(captures)
                    if person_count > 0:
                        logger.info(f"📸 Captured {person_count} person(s) from event {self.state.current_event_id}")
            except Exception as e:
                logger.warning(f"Person capture failed (non-critical): {e}")
        
        # Update event in database
        try:
            event = await EventRepository.finalize_event(
                event_id=self.state.current_event_id,
                end_time=end_time,
                scores=scores,
                frame_count=self.state.event_frame_count,
                clip_path=clip_path,
                clip_duration=clip_duration,
                thumbnail_path=thumbnail_path,
                person_images=person_image_filenames if person_image_filenames else None,
                person_count=person_count
            )
            
            if event:
                event_info = event.to_dict()
                
                # Notify callback
                if self.on_event_end:
                    try:
                        self.on_event_end(self.stream_id, event_info)
                    except Exception as e:
                        logger.error(f"Event end callback error: {e}")
                
                # Send alert
                if self.on_alert:
                    try:
                        self.on_alert(event_info)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
        
        except Exception as e:
            logger.error(f"Failed to finalize event: {e}")
        
        # Enter cooldown
        self.state.last_event_end = end_time
        self.state.cooldown_until = end_time + timedelta(seconds=self.cooldown_seconds)
        self.state.phase = DetectorState.Phase.COOLDOWN
        self.state.reset_event()
        
        logger.debug(f"Stream {self.stream_name}: Entered cooldown until {self.state.cooldown_until}")
    
    async def _log_inference(self, score: InferenceScore) -> None:
        """Log inference result to database (non-blocking)."""
        try:
            await InferenceLogRepository.create(
                stream_id=self.stream_id,
                violence_score=score.violence_score,
                non_violence_score=score.non_violence_score,
                inference_time_ms=int(score.inference_time_ms),
                frame_number=score.frame_count,
                window_start=score.window_start,
                window_end=score.window_end
            )
        except Exception as e:
            logger.debug(f"Failed to log inference: {e}")  # Non-critical
    
    def force_end_event(self) -> None:
        """Force end current event (used on stream stop)."""
        if self.state.phase in [DetectorState.Phase.ACTIVE, DetectorState.Phase.ENDING]:
            asyncio.create_task(self._finalize_event(InferenceScore(
                violence_score=0,
                non_violence_score=1,
                timestamp=datetime.utcnow(),
                inference_time_ms=0,
                frame_count=0,
                window_start=datetime.utcnow(),
                window_end=datetime.utcnow()
            )))
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "stream_id": self.stream_id,
            "stream_name": self.stream_name,
            "phase": self.state.phase.value,
            "threshold": self.threshold,
            "min_consecutive": self.min_consecutive,
            "cooldown_seconds": self.cooldown_seconds,
            "total_inferences": self.total_inferences,
            "total_events": self.total_events,
            "current_event_id": self.state.current_event_id,
            "state": self.state.to_dict()
        }


class MultiStreamEventManager:
    """
    Manages event detection across multiple streams.
    Provides unified alerting and statistics.
    """
    
    def __init__(self):
        self.detectors: Dict[str, EventDetectionEngine] = {}
        self._global_alert_callback: Optional[Callable[[Dict], None]] = None
    
    def set_global_alert_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set callback for all alerts from all streams."""
        self._global_alert_callback = callback
    
    def add_detector(
        self,
        stream_id: str,
        stream_name: str,
        ingestion: FFmpegIngestion,
        **kwargs
    ) -> EventDetectionEngine:
        """Create and register a new detector for a stream."""
        
        def on_alert(event_info: Dict):
            """Forward alerts to global handler."""
            if self._global_alert_callback:
                self._global_alert_callback(event_info)
        
        detector = EventDetectionEngine(
            stream_id=stream_id,
            stream_name=stream_name,
            ingestion=ingestion,
            on_alert=on_alert,
            **kwargs
        )
        
        self.detectors[stream_id] = detector
        return detector
    
    def remove_detector(self, stream_id: str) -> None:
        """Remove a detector."""
        if stream_id in self.detectors:
            self.detectors[stream_id].force_end_event()
            del self.detectors[stream_id]
    
    def get_detector(self, stream_id: str) -> Optional[EventDetectionEngine]:
        """Get detector by stream ID."""
        return self.detectors.get(stream_id)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all detectors."""
        return {
            stream_id: detector.stats
            for stream_id, detector in self.detectors.items()
        }
    
    def get_active_events(self) -> List[Dict]:
        """Get all currently active events."""
        active = []
        for stream_id, detector in self.detectors.items():
            if detector.state.current_event_id:
                active.append({
                    "stream_id": stream_id,
                    "stream_name": detector.stream_name,
                    "event_id": detector.state.current_event_id,
                    "start_time": detector.state.event_start_time.isoformat() if detector.state.event_start_time else None,
                    "peak_score": detector.state.peak_score,
                    "score_count": len(detector.state.event_scores)
                })
        return active


# Global instance
event_manager = MultiStreamEventManager()

__all__ = [
    "InferenceScore",
    "DetectorState",
    "EventDetectionEngine",
    "MultiStreamEventManager",
    "event_manager"
]
