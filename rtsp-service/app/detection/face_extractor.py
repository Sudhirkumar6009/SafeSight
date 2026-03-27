"""
Face Extraction Module for Violence Event Clips
================================================
Extracts and saves participant faces from violence event video clips.
Faces are stored as binary (BYTEA) in PostgreSQL.
"""

import cv2
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from loguru import logger
import threading


class FaceExtractor:
    """
    Extracts faces from video clips.
    
    Faces are extracted and returned as bytes for storage in PostgreSQL.
    No longer saves faces to filesystem.
    """
    
    def __init__(self):
        """Initialize the face extractor."""
        # Load Haar Cascade face detector
        # cv2.data.haarcascades provides the path to cascade files
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
        except AttributeError:
            # Fallback for some OpenCV installations
            cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade face detector")
            raise IOError("Failed to load Haar Cascade xml file.")
        
        # Detection parameters
        self.scale_factor = 1.1
        self.min_neighbors = 6  # Balanced for accuracy
        self.min_face_size = (50, 50)
        self.frame_interval = 0.5  # Extract frame every 0.5 seconds
        self.padding = 15  # Padding around detected face
        self.jpeg_quality = 90  # JPEG compression quality
        
        # Face deduplication - track face positions to avoid duplicates
        self.min_face_distance = 50  # Minimum pixel distance between "different" faces
        
        logger.info("FaceExtractor initialized (PostgreSQL binary storage)")
    
    def process_clip(self, clip_path: str, event_id: str = None) -> List[Dict[str, Any]]:
        """
        Process a video clip and extract unique faces as binary data.
        
        Args:
            clip_path: Path to the video clip file
            event_id: Optional event ID (for logging)
            
        Returns:
            List of face data dicts with keys:
            - image_data: bytes (JPEG)
            - face_index: int
            - bbox: tuple (x, y, w, h)
            - frame_number: int
            - frame_timestamp_ms: int
        """
        if not os.path.exists(clip_path):
            logger.warning(f"Clip not found: {clip_path}")
            return []
        
        logger.info(f"Extracting faces from clip: {clip_path}")
        
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.error(f"Failed to open clip: {clip_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        frame_skip = max(1, int(fps * self.frame_interval))
        
        extracted_faces: List[Dict[str, Any]] = []
        seen_faces: List[Tuple[int, int, int, int]] = []  # Track face positions
        frame_count = 0
        face_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                faces = self._detect_faces(frame)
                frame_timestamp_ms = int((frame_count / fps) * 1000)
                
                for (x, y, w, h) in faces:
                    # Check if this is a new face (not too close to previously seen faces)
                    if self._is_new_face(x, y, w, h, seen_faces):
                        face_data = self._extract_face_data(
                            frame, x, y, w, h, 
                            face_index, frame_count, frame_timestamp_ms
                        )
                        if face_data:
                            extracted_faces.append(face_data)
                            seen_faces.append((x, y, w, h))
                            face_index += 1
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(extracted_faces)} unique faces from clip")
        return extracted_faces
    
    def process_clip_for_db(
        self, 
        clip_path: str, 
        clip_id: int,
        event_id: int,
        stream_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process clip and return face data ready for database storage.
        
        Args:
            clip_path: Path to the video clip file
            clip_id: Database ID of the VideoClip
            event_id: Database ID of the Event
            stream_id: Optional database ID of the Stream
            
        Returns:
            List of face data dicts ready for StorageService.save_faces_batch()
        """
        faces = self.process_clip(clip_path)
        
        # Add IDs to each face record
        for face in faces:
            face['clip_id'] = clip_id
            face['event_id'] = event_id
            face['stream_id'] = stream_id
        
        return faces
    
    def process_clip_async(self, clip_path: str, event_id: str, callback=None):
        """
        Process clip in background thread.
        
        Args:
            clip_path: Path to the video clip
            event_id: Event ID (for logging)
            callback: Optional callback function(event_id, face_data_list)
        """
        def _process():
            try:
                faces = self.process_clip(clip_path, event_id)
                if callback:
                    callback(event_id, faces)
            except Exception as e:
                logger.error(f"Face extraction failed for {event_id}: {e}")
                if callback:
                    callback(event_id, [])
        
        thread = threading.Thread(target=_process, daemon=True)
        thread.start()
        return thread
    
    def _detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Improve contrast
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of tuples
            result = []
            for face in faces:
                result.append((int(face[0]), int(face[1]), int(face[2]), int(face[3])))
            return result
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _is_new_face(self, x: int, y: int, w: int, h: int, 
                      seen_faces: List[Tuple[int, int, int, int]]) -> bool:
        """Check if this face is sufficiently different from previously seen faces."""
        center_x = x + w // 2
        center_y = y + h // 2
        
        for (sx, sy, sw, sh) in seen_faces:
            seen_cx = sx + sw // 2
            seen_cy = sy + sh // 2
            
            distance = ((center_x - seen_cx) ** 2 + (center_y - seen_cy) ** 2) ** 0.5
            
            # If too close and similar size, probably same person
            size_ratio = (w * h) / max(1, sw * sh)
            if distance < self.min_face_distance and 0.5 < size_ratio < 2.0:
                return False
        
        return True
    
    def _extract_face_data(
        self, 
        frame, 
        x: int, 
        y: int, 
        w: int, 
        h: int,
        face_index: int,
        frame_number: int,
        frame_timestamp_ms: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract face region from frame and return as binary JPEG data.
        
        Returns:
            Dict with image_data, face_index, bbox, frame_number, frame_timestamp_ms
            or None on failure
        """
        try:
            # Add padding around face
            x_start = max(0, x - self.padding)
            y_start = max(0, y - self.padding)
            x_end = min(frame.shape[1], x + w + self.padding)
            y_end = min(frame.shape[0], y + h + self.padding)
            
            face_roi = frame[y_start:y_end, x_start:x_end]
            
            if face_roi.size == 0:
                return None
            
            # Encode as JPEG bytes
            success, buffer = cv2.imencode(
                '.jpg', 
                face_roi, 
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            
            if not success:
                return None
            
            return {
                'image_data': buffer.tobytes(),
                'face_index': face_index,
                'bbox': (x, y, w, h),
                'frame_number': frame_number,
                'frame_timestamp_ms': frame_timestamp_ms,
                'confidence': None  # Haar cascade doesn't provide confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to extract face: {e}")
            return None
    
    def _save_face(self, frame, x: int, y: int, w: int, h: int, 
                   output_dir: Path, face_index: int) -> Optional[Path]:
        """
        LEGACY: Extract and save a face region to filesystem.
        Kept for backward compatibility, but prefer _extract_face_data for new code.
        """
        try:
            # Add padding around face
            x_start = max(0, x - self.padding)
            y_start = max(0, y - self.padding)
            x_end = min(frame.shape[1], x + w + self.padding)
            y_end = min(frame.shape[0], y + h + self.padding)
            
            face_roi = frame[y_start:y_end, x_start:x_end]
            
            if face_roi.size == 0:
                return None
            
            # Generate filename with timestamp
            ts = datetime.now().strftime("%H%M%S%f")[:10]
            filename = f"participant_{face_index:02d}_{ts}.jpg"
            save_path = output_dir / filename
            
            # Save with good quality
            cv2.imwrite(str(save_path), face_roi, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            logger.debug(f"   Saved face: {filename}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save face: {e}")
            return None
    
    def extract_faces_from_frames(
        self, 
        frames: List[Tuple],  # List of (frame: np.ndarray, timestamp: float)
    ) -> List[Dict[str, Any]]:
        """
        Extract faces directly from in-memory frames.
        
        Args:
            frames: List of (frame, timestamp) tuples
            
        Returns:
            List of face data dicts with image_data, face_index, bbox, etc.
        """
        extracted_faces: List[Dict[str, Any]] = []
        seen_faces: List[Tuple[int, int, int, int]] = []
        face_index = 0
        
        # Calculate frame skip based on number of frames
        total_frames = len(frames)
        if total_frames == 0:
            return []
        
        # Estimate FPS from timestamps
        if total_frames >= 2:
            elapsed = frames[-1][1] - frames[0][1]
            fps = total_frames / elapsed if elapsed > 0 else 30
        else:
            fps = 30
        
        frame_skip = max(1, int(fps * self.frame_interval))
        
        for i, (frame, timestamp) in enumerate(frames):
            if i % frame_skip == 0:
                detected = self._detect_faces(frame)
                frame_timestamp_ms = int(timestamp * 1000)
                
                for (x, y, w, h) in detected:
                    if self._is_new_face(x, y, w, h, seen_faces):
                        face_data = self._extract_face_data(
                            frame, x, y, w, h,
                            face_index, i, frame_timestamp_ms
                        )
                        if face_data:
                            extracted_faces.append(face_data)
                            seen_faces.append((x, y, w, h))
                            face_index += 1
        
        logger.info(f"Extracted {len(extracted_faces)} faces from {total_frames} frames")
        return extracted_faces


# Global instance
face_extractor: Optional[FaceExtractor] = None


def get_face_extractor() -> FaceExtractor:
    """Get or create the global face extractor instance."""
    global face_extractor
    if face_extractor is None:
        face_extractor = FaceExtractor()
    return face_extractor
