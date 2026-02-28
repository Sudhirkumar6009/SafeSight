"""
SafeSight - Person Capture Module
======================================
Detects and crops individual persons from violence event frames
using OpenCV's HOG person detector + Haar Cascade face detector.
Saves cropped person images alongside event clips.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class PersonCapture:
    """Represents a captured person from a violence event."""
    person_index: int
    image_path: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    frame_timestamp: Optional[datetime] = None


class PersonCaptureEngine:
    """
    Detects and captures images of persons involved in violence events.
    Uses a combination of:
    1. HOG + SVM full-body person detector
    2. Haar Cascade face detector as fallback
    """

    def __init__(self, clips_dir: str = "./clips"):
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Parameters
        self.min_person_height = 80   # Minimum person height in pixels
        self.min_face_size = 30       # Minimum face size in pixels
        self.padding_ratio = 0.15     # Padding around detected person
        self.max_persons = 6          # Maximum persons to capture per event
        self.nms_threshold = 0.4      # Non-max suppression threshold

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in a frame using HOG descriptor.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        # Resize frame for faster detection if too large
        h, w = frame.shape[:2]
        scale = 1.0
        if w > 800:
            scale = 800.0 / w
            frame_resized = cv2.resize(frame, (800, int(h * scale)))
        else:
            frame_resized = frame

        # Detect persons using HOG
        boxes, weights = self.hog.detectMultiScale(
            frame_resized,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )

        if len(boxes) == 0:
            return []

        # Scale boxes back to original size
        persons = []
        for (x, y, bw, bh), weight in zip(boxes, weights):
            if weight < 0.3:  # Filter low confidence
                continue
            x = int(x / scale)
            y = int(y / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)
            if bh >= self.min_person_height:
                persons.append((x, y, bw, bh))

        # Apply non-maximum suppression
        if persons:
            persons = self._non_max_suppression(persons, self.nms_threshold)

        return persons[:self.max_persons]

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using Haar Cascade.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        if len(faces) == 0:
            return []

        return [tuple(f) for f in faces[:self.max_persons]]

    def _expand_face_to_upper_body(
        self, face_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        """Expand a face bounding box to include upper body (shoulders area)."""
        x, y, w, h = face_bbox
        frame_h, frame_w = frame_shape[:2]

        # Expand: wider by 2x, extend downward by 3x face height
        new_w = int(w * 2.5)
        new_h = int(h * 4)
        new_x = max(0, x - int(w * 0.75))
        new_y = max(0, y - int(h * 0.3))

        # Clamp to frame
        new_w = min(new_w, frame_w - new_x)
        new_h = min(new_h, frame_h - new_y)

        return (new_x, new_y, new_w, new_h)

    def _non_max_suppression(
        self, boxes: List[Tuple[int, int, int, int]], threshold: float
    ) -> List[Tuple[int, int, int, int]]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not boxes:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
        y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
        areas = boxes_arr[:, 2] * boxes_arr[:, 3]

        # Sort by area (larger first)
        order = areas.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            intersection = inter_w * inter_h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            remaining = np.where(iou <= threshold)[0]
            order = order[remaining + 1]

        return [boxes[i] for i in keep]

    def _add_padding(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...],
        padding_ratio: float = None
    ) -> Tuple[int, int, int, int]:
        """Add padding around a bounding box, clamped to frame size."""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        pad = padding_ratio or self.padding_ratio

        pad_x = int(w * pad)
        pad_y = int(h * pad)

        new_x = max(0, x - pad_x)
        new_y = max(0, y - pad_y)
        new_w = min(w + 2 * pad_x, frame_w - new_x)
        new_h = min(h + 2 * pad_y, frame_h - new_y)

        return (new_x, new_y, new_w, new_h)

    def capture_persons_from_frames(
        self,
        frames: List[np.ndarray],
        event_id: str,
        stream_id: str,
    ) -> List[PersonCapture]:
        """
        Detect and capture persons from event frames.

        Strategy:
        1. Try HOG person detection on key frames (start, peak, middle)
        2. If < 2 persons found, fall back to face detection
        3. Crop and save unique person images
        """
        if not frames:
            return []

        captures: List[PersonCapture] = []
        best_detections: List[Tuple[Tuple[int, int, int, int], np.ndarray, float]] = []

        # Sample key frames: beginning, 1/3, 1/2, 2/3 of the clip
        num_frames = len(frames)
        sample_indices = list(set([
            0,
            num_frames // 4,
            num_frames // 3,
            num_frames // 2,
            2 * num_frames // 3,
            min(num_frames - 1, num_frames - 5),
        ]))
        sample_indices = [i for i in sample_indices if 0 <= i < num_frames]

        # Try HOG person detection on sampled frames
        for idx in sample_indices:
            frame = frames[idx]
            persons = self.detect_persons(frame)

            if len(persons) >= 2:
                # Found 2+ persons — use this frame
                for bbox in persons:
                    padded = self._add_padding(bbox, frame.shape)
                    x, y, w, h = padded
                    crop = frame[y:y+h, x:x+w].copy()
                    if crop.size > 0:
                        best_detections.append((padded, crop, 0.9))
                break  # Use first frame with >=2 persons

        # If HOG didn't find enough, try face detection
        if len(best_detections) < 2:
            for idx in sample_indices:
                frame = frames[idx]
                faces = self.detect_faces(frame)

                if len(faces) >= 2:
                    best_detections = []  # Reset
                    for face_bbox in faces:
                        # Expand face to upper body
                        body_bbox = self._expand_face_to_upper_body(face_bbox, frame.shape)
                        padded = self._add_padding(body_bbox, frame.shape, 0.05)
                        x, y, w, h = padded
                        crop = frame[y:y+h, x:x+w].copy()
                        if crop.size > 0:
                            best_detections.append((padded, crop, 0.7))
                    break

        # If still not enough, try face detection for individual faces
        if len(best_detections) < 2:
            for idx in sample_indices:
                frame = frames[idx]
                faces = self.detect_faces(frame)
                for face_bbox in faces:
                    padded = self._add_padding(face_bbox, frame.shape, 0.4)
                    x, y, w, h = padded
                    crop = frame[y:y+h, x:x+w].copy()
                    if crop.size > 0 and len(best_detections) < self.max_persons:
                        best_detections.append((padded, crop, 0.5))
                if len(best_detections) >= 2:
                    break

        # Save captured person images
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        for i, (bbox, crop, confidence) in enumerate(best_detections[:self.max_persons]):
            filename = f"{stream_id}_{event_id}_person{i+1}_{timestamp}.jpg"
            img_path = self.clips_dir / filename

            try:
                # Resize crop for consistent output (max 300px height)
                crop_h, crop_w = crop.shape[:2]
                if crop_h > 300:
                    scale = 300.0 / crop_h
                    crop = cv2.resize(crop, (int(crop_w * scale), 300))

                cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

                captures.append(PersonCapture(
                    person_index=i + 1,
                    image_path=str(img_path),
                    bbox=bbox,
                    confidence=confidence,
                ))
                logger.info(f"Captured person {i+1} for event {event_id}: {filename}")

            except Exception as e:
                logger.error(f"Failed to save person capture {i+1}: {e}")

        logger.info(
            f"Person capture complete for event {event_id}: "
            f"{len(captures)} person(s) detected"
        )
        return captures


# Global instance
person_capture_engine = PersonCaptureEngine()

__all__ = ["PersonCapture", "PersonCaptureEngine", "person_capture_engine"]
