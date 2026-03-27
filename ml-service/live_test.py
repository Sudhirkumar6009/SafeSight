#!/usr/bin/env python3
"""
SafeSight ML Service - Live Webcam Test
========================================
Test the ONNX violence detector with your webcam.
Based on trusted reference implementation.

Usage:
    python live_test.py                    # Default webcam (0)
    python live_test.py --source 1         # Different camera index
    python live_test.py --source video.mp4 # Video file

Controls:
    q - Quit
    r - Reset frame queue
"""

import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
import argparse
import sys

# ==========================================
# CONFIGURATION - from trusted reference
# ==========================================
ONNX_MODEL_PATH = "./models/violence_model.onnx"

# Class labels - from trusted reference
CLASS_0 = "Normal"
CLASS_1 = "Violence"

# Model parameters - from trusted reference
SEQ_LEN = 16
IMG_SIZE = 224
FRAME_SKIP = 3  # Grab every 3rd frame so 16 frames = ~1.5 seconds of real time


def main():
    parser = argparse.ArgumentParser(description="Test ONNX violence detector with webcam")
    parser.add_argument("--source", default="0", help="Video source: camera index or file path")
    parser.add_argument("--model", default=ONNX_MODEL_PATH, help="Path to ONNX model")
    args = parser.parse_args()

    # ==========================================
    # LOAD THE ONNX ENGINE - from trusted reference
    # ==========================================
    print(f"Loading ONNX Engine from: {args.model}...")
    try:
        session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        print("Model loaded! Warming up webcam...\n")
    except Exception as e:
        print(f"Failed to load ONNX model. Error: {e}")
        return 1

    # ==========================================
    # LIVE WEBCAM STREAM - from trusted reference
    # ==========================================
    # Initialize webcam (0 is usually the built-in laptop cam, 1 is a USB cam)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return 1

    # The "Conveyer Belt" that holds our 16 frames - from trusted reference
    frame_queue = deque(maxlen=SEQ_LEN)
    frame_counter = 0
    current_label = "Waiting for data..."
    current_color = (0, 255, 255)  # Yellow

    print("Live Stream Started! Press 'q' on your keyboard to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                # Video file ended, loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Failed to grab frame from webcam. Is it being used by another app?")
                break

        # 1. Grab every Nth frame for the AI's memory - from trusted reference
        frame_counter += 1
        if frame_counter % FRAME_SKIP == 0:
            # Pre-process for the AI - from trusted reference
            ai_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
            ai_frame = np.array(ai_frame, dtype=np.float32) / 255.0

            # Add to our rolling window
            frame_queue.append(ai_frame)

        # 2. If our conveyer belt is full (16 frames), run a prediction! - from trusted reference
        if len(frame_queue) == SEQ_LEN:
            # Create the exact batch shape the ONNX model expects: (1, 16, 224, 224, 3)
            input_tensor = np.expand_dims(np.array(frame_queue), axis=0)

            # Fire the ONNX Engine - from trusted reference
            outputs = session.run(None, {input_name: input_tensor})
            prediction_score = outputs[0][0][0]
            confidence = prediction_score * 100

            # Update the screen text based on the AI's thoughts - from trusted reference
            if prediction_score > 0.50:
                current_label = f"[!] {CLASS_1} ({confidence:.1f}%)"
                current_color = (0, 0, 255)  # Red for danger
            else:
                normal_confidence = 100 - confidence
                current_label = f"[OK] {CLASS_0} ({normal_confidence:.1f}%)"
                current_color = (0, 255, 0)  # Green for safe

        # 3. Draw the AI's thoughts directly onto the live webcam feed - from trusted reference
        cv2.putText(frame, current_label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_color, 3)

        # Show queue status
        queue_status = f"Queue: {len(frame_queue)}/{SEQ_LEN}"
        cv2.putText(frame, queue_status, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the video window
        cv2.imshow("SafeSight Violence Detection - Live View", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_queue.clear()
            frame_counter = 0
            current_label = "Waiting for data..."
            current_color = (0, 255, 255)
            print("Frame queue reset")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Live Stream Closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
