"""
RTSP Live Stream Service - Event Detection
===========================================
Threshold + duration rules to create real alerts.

Detection flow:
  1. Each inference result is checked against violence_threshold
  2. After min_consecutive violent frames → event starts 
  3. Pre-event frames (10s) are snapshotted for clips
  4. After 15s → quick alert clip saved (5s pre + 15s post), notification sent
  5. When violence stops (consecutive non-violent frames) → event ends
  6. Full evidence clip saved (10s pre + violence duration + 10s post)
  7. Short cooldown, then ready for next event
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.config import settings
from app.database import Event, EventStatus, AlertSeverity, InferenceLog, async_session
from app.inference.pipeline import InferenceResult, InferencePipeline
from app.stream.ingestion import StreamIngestion, ClipRecorder, FrameData


@dataclass
class EventState:
    """Tracks the state of an ongoing potential event."""
    stream_id: int
    stream_name: str
    is_active: bool = False
    start_time: Optional[datetime] = None
    scores: List[float] = field(default_factory=list)
    frame_buffer: List[FrameData] = field(default_factory=list)
    pre_event_frames: List[FrameData] = field(default_factory=list)
    consecutive_high_scores: int = 0
    consecutive_low_scores: int = 0        # Track how many non-violent frames in a row
    last_high_score_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    alert_clip_saved: bool = False
    
    # High-confidence (90%+) clip tracking
    high_conf_active: bool = False
    high_conf_start_time: Optional[datetime] = None
    high_conf_end_time: Optional[datetime] = None
    high_conf_pre_frames: List[FrameData] = field(default_factory=list)
    high_conf_peak_score: float = 0.0


class EventDetector:
    """
    Detects violence events based on inference results.
    Uses threshold + consecutive frame rules to generate alerts.
    """
    
    # How many consecutive non-violent frames to end an event
    END_CONSECUTIVE = 3  # At 5 FPS = 0.6s of calm needed to end
    
    def __init__(
        self,
        stream: StreamIngestion,
        pipeline: InferencePipeline,
        on_event_start: Optional[Callable[[int, Dict], None]] = None,
        on_event_end: Optional[Callable[[int, Dict], None]] = None,
        on_alert: Optional[Callable[[Event], None]] = None
    ):
        self.stream = stream
        self.pipeline = pipeline
        self.on_event_start = on_event_start
        self.on_event_end = on_event_end
        self.on_alert = on_alert
        
        self.clip_recorder = ClipRecorder(settings.clips_dir)
        
        # State
        self.state = EventState(
            stream_id=stream.config.id,
            stream_name=stream.config.name
        )
        
        # Configuration
        self.threshold = settings.violence_threshold
        self.clip_conf_threshold = settings.clip_confidence_threshold  # 90% for clip recording
        self.min_consecutive = settings.min_consecutive_frames
        self.cooldown_seconds = settings.alert_cooldown_seconds
        self.clip_before = settings.clip_duration_before      # 5s for quick alert clip
        self.clip_after = settings.clip_duration_after        # 15s for quick alert clip  
        self.full_clip_before = settings.full_clip_before     # 10s for full evidence clip
        self.full_clip_after = settings.full_clip_after       # 10s for full evidence clip
        
        # Event tracking
        self.current_event_id: Optional[int] = None
        self.pending_end_task: Optional[asyncio.Task] = None
        self.alert_clip_task: Optional[asyncio.Task] = None
        self._inference_count: int = 0
    
    async def process_result(self, result: InferenceResult):
        """Process an inference result and check for events.
        
        Uses camera shake detection and score stabilization to prevent
        false positives from rapid camera movement, static scenes,
        and suspicious motion.
        
        CRITICAL: Camera movement causes 98-100% false positives from the model.
        We ONLY trigger events when the camera is stable.
        """
        now = datetime.utcnow()
        self._inference_count += 1
        
        # Always log inference (even during cooldown)
        await self._log_inference(result)
        
        # Check cooldown — but only block event creation, not scoring
        in_cooldown = self.state.cooldown_until and now < self.state.cooldown_until
        
        # Get all motion analysis flags
        is_camera_shake = getattr(result, 'is_camera_shake', False)
        is_stable = getattr(result, 'is_stable', True)
        is_confirmed = getattr(result, 'is_confirmed', False)
        stabilized_score = getattr(result, 'stabilized_score', result.violence_score)
        raw_score = getattr(result, 'raw_score', result.violence_score)
        
        # CRITICAL: Any camera motion = reject
        # is_stable means camera has been stable for 2+ seconds
        is_problematic = is_camera_shake or not is_stable
        
        # Violence is detected ONLY when camera is stable
        is_violent = result.is_violent  # Uses the updated property with stability handling
        
        # During any camera motion, NEVER trigger
        if is_problematic:
            is_violent = False  # Absolute rejection during camera movement
            if raw_score >= self.threshold:
                logger.debug(
                    f"🚫 Rejected on {self.state.stream_name}: "
                    f"raw={raw_score:.1%} stable={is_stable} shake={is_camera_shake}"
                )
        
        if is_violent:
            self.state.consecutive_high_scores += 1
            self.state.consecutive_low_scores = 0
            self.state.last_high_score_time = now
            
            # === HIGH-CONFIDENCE CLIP TRACKING (90%+ threshold) ===
            actual_score = result.violence_score
            if actual_score >= self.clip_conf_threshold:
                if not self.state.high_conf_active:
                    # First hit of 90%+ - start high-confidence period
                    self.state.high_conf_active = True
                    self.state.high_conf_start_time = now
                    self.state.high_conf_end_time = None
                    # Snapshot 10s of pre-event frames for the clip
                    self.state.high_conf_pre_frames = self.stream.get_frame_window(self.full_clip_before)
                    logger.warning(
                        f"🎬 HIGH-CONF clip started on {self.state.stream_name} "
                        f"(score: {actual_score:.2%}, buffered {len(self.state.high_conf_pre_frames)} pre-frames)"
                    )
                # Track peak score
                if actual_score > self.state.high_conf_peak_score:
                    self.state.high_conf_peak_score = actual_score
            
            if self.state.is_active:
                # === ONGOING EVENT: accumulate scores ===
                self.state.scores.append(result.violence_score)
                
                # Cancel any pending end task — violence resumed
                if self.pending_end_task and not self.pending_end_task.done():
                    self.pending_end_task.cancel()
                    self.pending_end_task = None
                    logger.debug(f"Violence resumed on {self.state.stream_name}, cancelled event end")
                    
            elif not in_cooldown:
                # === NOT IN EVENT: check if we should start one ===
                # Prefer confirmed violence (sustained 4+ seconds), but also allow
                # immediate trigger on very high confidence non-problematic frames
                should_start_event = False
                
                if is_confirmed and is_stable:
                    # Sustained violence confirmed AND camera is stable
                    should_start_event = True
                    logger.info(f"Starting event due to CONFIRMED violence on {self.state.stream_name}")
                elif self.state.consecutive_high_scores >= self.min_consecutive and is_stable:
                    # Enough consecutive high scores AND camera is stable
                    if not is_problematic and stabilized_score >= self.threshold:
                        should_start_event = True
                        logger.info(f"Starting event due to {self.state.consecutive_high_scores} consecutive frames")
                    elif stabilized_score >= 0.85:
                        # Very high confidence - camera must still be stable
                        should_start_event = True
                        logger.info(f"Starting event due to HIGH CONFIDENCE ({stabilized_score:.1%})")
                
                if should_start_event:
                    await self._start_event(result)
        else:
            # Non-violent frame
            self.state.consecutive_low_scores += 1
            
            # === HIGH-CONFIDENCE CLIP END DETECTION ===
            # Check if high-conf period dropped below 90%
            actual_score = result.violence_score
            if self.state.high_conf_active and actual_score < self.clip_conf_threshold:
                self.state.high_conf_end_time = now
                high_conf_duration = (now - self.state.high_conf_start_time).total_seconds() if self.state.high_conf_start_time else 0
                logger.warning(
                    f"🎬 HIGH-CONF clip ended on {self.state.stream_name} "
                    f"(duration: {high_conf_duration:.1f}s, peak: {self.state.high_conf_peak_score:.2%})"
                )
                # Schedule clip save after capturing post-event footage
                asyncio.create_task(self._save_high_conf_clip(high_conf_duration))
                # Reset high-conf tracking
                self.state.high_conf_active = False
            
            if self.state.is_active:
                # Still in event — add score for tracking
                self.state.scores.append(result.violence_score)
                
                # Check if enough consecutive non-violent frames to end
                if self.state.consecutive_low_scores >= self.END_CONSECUTIVE:
                    # Schedule event end (with post-event recording delay)
                    if not self.pending_end_task or self.pending_end_task.done():
                        self.pending_end_task = asyncio.create_task(
                            self._delayed_event_end(result)
                        )
            else:
                # Not in event — reset consecutive count
                self.state.consecutive_high_scores = 0
    
    async def _start_event(self, result: InferenceResult):
        """Start a new violence event."""
        self.state.is_active = True
        self.state.start_time = datetime.utcnow()
        self.state.scores = [result.violence_score]
        self.state.alert_clip_saved = False
        self.state.consecutive_low_scores = 0
        
        # Snapshot pre-event frames (10s for full evidence clip)
        self.state.pre_event_frames = self.stream.get_frame_window(self.full_clip_before)
        
        logger.warning(
            f"🚨 Violence event STARTED on {self.state.stream_name} "
            f"(score: {result.violence_score:.2%}, "
            f"buffered {len(self.state.pre_event_frames)} pre-event frames)"
        )
        
        # Create event in database
        event = await self._create_event(result)
        self.current_event_id = event.id if event else None
        
        # Broadcast event_start via WebSocket
        if self.on_event_start:
            self.on_event_start(self.state.stream_id, {
                "type": "event_start",
                "event_id": self.current_event_id,
                "stream_name": self.state.stream_name,
                "confidence": result.violence_score,
                "timestamp": self.state.start_time.isoformat(),
                "message": f"Violence detected on {self.state.stream_name}!"
            })
        
        # Schedule quick alert clip (5s before + 15s after = 20s total)
        if self.alert_clip_task and not self.alert_clip_task.done():
            self.alert_clip_task.cancel()
        self.alert_clip_task = asyncio.create_task(self._save_alert_clip())
    
    async def _save_alert_clip(self):
        """
        Save a quick 20s alert clip (5s before + 15s after violence start).
        This clip is saved quickly for immediate notification.
        """
        try:
            # Wait clip_after seconds to capture post-violence footage
            await asyncio.sleep(self.clip_after)
            
            if not self.current_event_id:
                return
            
            # Get pre-event frames (last 5s from the snapshot)
            pre_frames = self.state.pre_event_frames or []
            # Only take last clip_before seconds worth of frames
            fps = self.stream.config.target_fps or 30
            pre_count = self.clip_before * fps
            if len(pre_frames) > pre_count:
                pre_frames = pre_frames[-pre_count:]
            
            # Get post-event frames (clip_after seconds of footage after start)
            post_frames = self.stream.get_frame_window(self.clip_after)
            
            # Combine and deduplicate
            seen = set()
            alert_frames = []
            for f in pre_frames:
                if f.frame_number not in seen:
                    seen.add(f.frame_number)
                    alert_frames.append(f)
            for f in post_frames:
                if f.frame_number not in seen:
                    seen.add(f.frame_number)
                    alert_frames.append(f)
            
            alert_frames.sort(key=lambda f: f.frame_number)
            
            # Calculate real duration from timestamps
            if len(alert_frames) >= 2:
                elapsed = (alert_frames[-1].timestamp - alert_frames[0].timestamp).total_seconds()
                alert_duration = elapsed if elapsed > 0 else len(alert_frames) / 30.0
            else:
                alert_duration = len(alert_frames) / 30.0
            
            logger.info(
                f"📹 Alert clip: {len(alert_frames)} frames (~{alert_duration:.1f}s) "
                f"for event {self.current_event_id}"
            )
            
            if alert_frames:
                clip_path = self.clip_recorder.save_clip(
                    alert_frames,
                    self.state.stream_name,
                    self.current_event_id,
                    suffix="_alert"
                )
                
                # Save thumbnail
                peak_idx = len(pre_frames) if pre_frames else len(alert_frames) // 2
                peak_idx = min(peak_idx, len(alert_frames) - 1)
                thumbnail_path = self.clip_recorder.save_thumbnail(
                    alert_frames[peak_idx].frame,
                    self.state.stream_name,
                    self.current_event_id
                )
                
                # Update event with alert clip
                from pathlib import Path as P
                clip_filename = P(clip_path).name if clip_path else None
                thumb_filename = P(thumbnail_path).name if thumbnail_path else None
                
                await self._update_event_clip(
                    self.current_event_id,
                    clip_filename,
                    thumb_filename,
                    alert_duration
                )
                
                self.state.alert_clip_saved = True
                
                # Broadcast alert notification with clip info
                if self.on_alert:
                    self.on_alert(None)  # Trigger notification
                
                # Send via WebSocket (on_event_end callback will be called later for full clip)
                if self.on_event_start:
                    self.on_event_start(self.state.stream_id, {
                        "type": "violence_alert",
                        "event_id": self.current_event_id,
                        "stream_name": self.state.stream_name,
                        "clip_path": clip_filename,
                        "thumbnail_path": thumb_filename,
                        "clip_duration": alert_duration,
                        "max_confidence": max(self.state.scores) if self.state.scores else 0,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": f"Violence detected on {self.state.stream_name}!"
                    })
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Failed to save alert clip: {e}")
    
    async def _update_event_clip(
        self, event_id: int, clip_path: str, thumbnail_path: str, clip_duration: float
    ):
        """Update event with clip info (alert or full)."""
        try:
            async with async_session() as session:
                values = {}
                if clip_path:
                    values["clip_path"] = clip_path
                if thumbnail_path:
                    values["thumbnail_path"] = thumbnail_path
                if clip_duration:
                    values["clip_duration"] = clip_duration
                
                if values:
                    await session.execute(
                        update(Event)
                        .where(Event.id == event_id)
                        .values(**values)
                    )
                    await session.commit()
        except Exception as e:
            logger.error(f"Failed to update event clip: {e}")
    
    async def _save_high_conf_clip(self, violence_duration: float):
        """
        Save a full evidence clip for high-confidence (90%+) violence period.
        
        Structure: 10s before + violence duration + 10s after = complete evidence
        Example: 30s violence → 10 + 30 + 10 = 50s clip
        """
        try:
            # Wait for 10s of post-violence footage
            await asyncio.sleep(self.full_clip_after)
            
            pre_frames = self.state.high_conf_pre_frames or []
            
            # Get all frames from violence start to now (includes violence + post)
            total_window = violence_duration + self.full_clip_after + 2  # +2s buffer
            post_frames = self.stream.get_frame_window(total_window)
            
            # Combine and deduplicate frames
            seen_frame_numbers = set()
            all_frames = []
            
            for f in pre_frames:
                if f.frame_number not in seen_frame_numbers:
                    seen_frame_numbers.add(f.frame_number)
                    all_frames.append(f)
            
            for f in post_frames:
                if f.frame_number not in seen_frame_numbers:
                    seen_frame_numbers.add(f.frame_number)
                    all_frames.append(f)
            
            all_frames.sort(key=lambda f: f.frame_number)
            
            if not all_frames:
                logger.warning("No frames available for high-conf clip")
                return
            
            # Calculate duration from timestamps
            if len(all_frames) >= 2:
                elapsed = (all_frames[-1].timestamp - all_frames[0].timestamp).total_seconds()
                clip_duration = elapsed if elapsed > 0 else len(all_frames) / 30.0
            else:
                clip_duration = len(all_frames) / 30.0
            expected_duration = self.full_clip_before + violence_duration + self.full_clip_after
            
            logger.info(
                f"📹 HIGH-CONF evidence clip: {len(pre_frames)} pre + violence({violence_duration:.1f}s) + post "
                f"= {len(all_frames)} frames (~{clip_duration:.1f}s, expected: {expected_duration:.1f}s)"
            )
            
            # Generate unique event ID for this clip (use timestamp if no event)
            clip_event_id = self.current_event_id or int(datetime.utcnow().timestamp())
            
            # Save the full evidence clip
            clip_path = self.clip_recorder.save_clip(
                all_frames,
                self.state.stream_name,
                clip_event_id,
                suffix="_evidence"
            )
            
            # Save thumbnail from peak violence moment (middle of violence period)
            peak_idx = len(pre_frames) + int(len(all_frames) * 0.3)  # ~30% into clip
            peak_idx = min(peak_idx, len(all_frames) - 1)
            thumbnail_path = self.clip_recorder.save_thumbnail(
                all_frames[peak_idx].frame,
                self.state.stream_name,
                clip_event_id
            )
            
            # Update event in database with evidence clip
            if self.current_event_id:
                await self._update_event_clip(
                    self.current_event_id,
                    clip_path.split('/')[-1] if clip_path else None,
                    thumbnail_path.split('/')[-1] if thumbnail_path else None,
                    clip_duration
                )
            
            # Broadcast notification
            if self.on_event_start:
                from pathlib import Path as P
                clip_filename = P(clip_path).name if clip_path else None
                thumb_filename = P(thumbnail_path).name if thumbnail_path else None
                
                self.on_event_start(self.state.stream_id, {
                    "type": "evidence_clip",
                    "event_id": clip_event_id,
                    "stream_name": self.state.stream_name,
                    "clip_path": clip_filename,
                    "thumbnail_path": thumb_filename,
                    "clip_duration": clip_duration,
                    "violence_duration": violence_duration,
                    "peak_confidence": self.state.high_conf_peak_score,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Evidence clip saved: {violence_duration:.0f}s violence @ {self.state.high_conf_peak_score:.0%}"
                })
            
            logger.warning(
                f"✅ HIGH-CONF evidence clip saved: {clip_path} "
                f"({clip_duration:.1f}s total, {violence_duration:.1f}s violence)"
            )
            
            # Reset peak score for next high-conf event
            self.state.high_conf_peak_score = 0.0
            self.state.high_conf_pre_frames = []
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Failed to save high-conf evidence clip: {e}")
    
    async def _delayed_event_end(self, final_result: InferenceResult):
        """Wait for full_clip_after seconds to capture post-event footage, then end."""
        try:
            logger.info(
                f"⏳ Violence stopped on {self.state.stream_name}, "
                f"waiting {self.full_clip_after}s for post-event footage..."
            )
            await asyncio.sleep(self.full_clip_after)
            
            # Only end if still active and still in low-score territory
            if self.state.is_active:
                await self._end_event(final_result)
                
        except asyncio.CancelledError:
            # Violence resumed — don't end event
            logger.debug(f"Event end cancelled — violence resumed on {self.state.stream_name}")
    
    async def _end_event(self, final_result: InferenceResult):
        """End the current violence event and save full evidence clip."""
        if not self.state.is_active:
            return
        
        end_time = datetime.utcnow()
        duration = (end_time - self.state.start_time).total_seconds() if self.state.start_time else 0
        
        avg_conf = sum(self.state.scores) / len(self.state.scores) if self.state.scores else 0
        max_conf = max(self.state.scores) if self.state.scores else 0
        
        logger.warning(
            f"✅ Violence event ENDED on {self.state.stream_name} "
            f"(duration: {duration:.1f}s, max: {max_conf:.2%}, avg: {avg_conf:.2%}, "
            f"frames: {len(self.state.scores)})"
        )
        
        # Cancel alert clip task if still running (should be done by now)
        if self.alert_clip_task and not self.alert_clip_task.done():
            self.alert_clip_task.cancel()
        
        # ===== FULL EVIDENCE CLIP =====
        # 10s before violence + entire violence duration + 10s after
        pre_frames = self.state.pre_event_frames or []
        post_frames = self.stream.get_frame_window(duration + self.full_clip_after)
        
        # Combine and deduplicate
        seen_frame_numbers = set()
        all_frames = []
        for f in pre_frames:
            if f.frame_number not in seen_frame_numbers:
                seen_frame_numbers.add(f.frame_number)
                all_frames.append(f)
        for f in post_frames:
            if f.frame_number not in seen_frame_numbers:
                seen_frame_numbers.add(f.frame_number)
                all_frames.append(f)
        
        all_frames.sort(key=lambda f: f.frame_number)
        
        # Calculate duration from timestamps
        if len(all_frames) >= 2:
            elapsed = (all_frames[-1].timestamp - all_frames[0].timestamp).total_seconds()
            full_clip_duration = elapsed if elapsed > 0 else len(all_frames) / 30.0
        else:
            full_clip_duration = len(all_frames) / 30.0
        
        logger.info(
            f"📹 Full evidence clip: {len(pre_frames)} pre + {len(post_frames)} post "
            f"= {len(all_frames)} total frames (~{full_clip_duration:.1f}s)"
        )
        
        # Save full evidence clip
        clip_path = None
        thumbnail_path = None
        
        if all_frames and self.current_event_id:
            clip_path = self.clip_recorder.save_clip(
                all_frames,
                self.state.stream_name,
                self.current_event_id,
                suffix="_full"
            )
            
            # Save thumbnail from peak violence moment
            peak_idx = len(pre_frames) if pre_frames else len(all_frames) // 2
            peak_idx = min(peak_idx, len(all_frames) - 1)
            thumbnail_path = self.clip_recorder.save_thumbnail(
                all_frames[peak_idx].frame,
                self.state.stream_name,
                self.current_event_id
            )
        
        # Update event in database with full evidence clip
        if self.current_event_id:
            await self._finalize_event(
                self.current_event_id,
                end_time,
                duration,
                clip_path,
                thumbnail_path,
                full_clip_duration
            )
        
        # Callback — broadcast event_end with full clip info
        if self.on_event_end:
            from pathlib import Path as P
            clip_filename = P(clip_path).name if clip_path else None
            thumb_filename = P(thumbnail_path).name if thumbnail_path else None
            
            self.on_event_end(self.state.stream_id, {
                "type": "event_end",
                "event_id": self.current_event_id,
                "stream_name": self.state.stream_name,
                "duration": duration,
                "clip_path": clip_filename,
                "clip_duration": full_clip_duration,
                "thumbnail_path": thumb_filename,
                "max_confidence": max_conf,
                "avg_confidence": avg_conf,
                "severity": self._calculate_severity(max_conf).value,
                "timestamp": end_time.isoformat(),
                "message": f"Violence event completed on {self.state.stream_name} ({duration:.0f}s) — Full clip recorded"
            })
        
        # === RESET STATE — ready for next event ===
        self._reset_state()
        logger.info(
            f"🔄 Detector reset for {self.state.stream_name} — "
            f"cooldown {self.cooldown_seconds}s, then ready for next event"
        )
    
    def _reset_state(self):
        """Reset all event state after an event ends."""
        self.state.is_active = False
        self.state.start_time = None
        self.state.scores = []
        self.state.frame_buffer = []
        self.state.pre_event_frames = []
        self.state.alert_clip_saved = False
        self.state.consecutive_high_scores = 0
        self.state.consecutive_low_scores = 0
        self.state.cooldown_until = datetime.utcnow() + timedelta(seconds=self.cooldown_seconds)
        self.current_event_id = None
        self.pending_end_task = None
        self.alert_clip_task = None
        # Reset high-confidence clip tracking
        self.state.high_conf_active = False
        self.state.high_conf_start_time = None
        self.state.high_conf_end_time = None
        self.state.high_conf_pre_frames = []
        self.state.high_conf_peak_score = 0.0
    
    async def _create_event(self, result: InferenceResult) -> Optional[Event]:
        """Create a new event in the database."""
        try:
            async with async_session() as session:
                event = Event(
                    stream_id=self.state.stream_id,
                    stream_name=self.state.stream_name,
                    start_time=self.state.start_time,
                    max_confidence=result.violence_score,
                    avg_confidence=result.violence_score,
                    min_confidence=result.violence_score,
                    frame_count=1,
                    severity=self._calculate_severity(result.violence_score),
                    status=EventStatus.PENDING
                )
                
                session.add(event)
                await session.commit()
                await session.refresh(event)
                
                return event
                
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return None
    
    async def _finalize_event(
        self,
        event_id: int,
        end_time: datetime,
        duration: float,
        clip_path: Optional[str],
        thumbnail_path: Optional[str],
        clip_duration: float = 0
    ):
        """Finalize event with clip and duration info."""
        try:
            async with async_session() as session:
                scores = self.state.scores
                
                # Extract just the filename from full path for portable storage
                clip_filename = None
                thumb_filename = None
                if clip_path:
                    from pathlib import Path as P
                    clip_filename = P(clip_path).name
                if thumbnail_path:
                    from pathlib import Path as P
                    thumb_filename = P(thumbnail_path).name
                
                await session.execute(
                    update(Event)
                    .where(Event.id == event_id)
                    .values(
                        end_time=end_time,
                        duration_seconds=duration,
                        max_confidence=max(scores) if scores else 0,
                        avg_confidence=sum(scores) / len(scores) if scores else 0,
                        min_confidence=min(scores) if scores else 0,
                        frame_count=len(scores),
                        clip_path=clip_filename,
                        clip_duration=clip_duration,
                        thumbnail_path=thumb_filename
                    )
                )
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to finalize event: {e}")
    
    async def _log_inference(self, result: InferenceResult):
        """Log inference result to database."""
        try:
            async with async_session() as session:
                log = InferenceLog(
                    stream_id=result.stream_id,
                    violence_score=result.violence_score,
                    non_violence_score=result.non_violence_score,
                    inference_time_ms=result.inference_time_ms,
                    frame_number=result.frame_count,
                    window_start=result.window_start,
                    window_end=result.window_end
                )
                session.add(log)
                await session.commit()
                
        except Exception as e:
            logger.debug(f"Failed to log inference: {e}")
    
    def _calculate_severity(self, confidence: float) -> AlertSeverity:
        """Calculate alert severity based on confidence."""
        if confidence >= 0.9:
            return AlertSeverity.CRITICAL
        elif confidence >= 0.8:
            return AlertSeverity.HIGH
        elif confidence >= 0.7:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            "stream_id": self.state.stream_id,
            "stream_name": self.state.stream_name,
            "is_active_event": self.state.is_active,
            "event_start_time": self.state.start_time.isoformat() if self.state.start_time else None,
            "consecutive_high_scores": self.state.consecutive_high_scores,
            "consecutive_low_scores": self.state.consecutive_low_scores,
            "current_event_id": self.current_event_id,
            "event_score_count": len(self.state.scores),
            "alert_clip_saved": self.state.alert_clip_saved,
            "in_cooldown": bool(self.state.cooldown_until and datetime.utcnow() < self.state.cooldown_until),
            "cooldown_until": self.state.cooldown_until.isoformat() if self.state.cooldown_until else None
        }
