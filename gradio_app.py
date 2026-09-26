"""Gradio Web Application for ASL Fingerspelling Recognition.
Optimized for deployment on Hugging Face Spaces with real-time WebSocket streaming.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2 as cv
import gradio as gr
import mediapipe as mp
import numpy as np

from gesture_text import GestureTextComposer, TemporalPredictionFilter
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# Hugging Face ZeroGPU compatibility hook
try:
    import spaces

    @spaces.GPU
    def _hf_zerogpu_startup():
        return True
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "keypoint_classifier" / "keypoint_classifier.tflite"
LABEL_PATH = PROJECT_ROOT / "model" / "keypoint_classifier" / "keypoint_classifier_label.csv"
CHART_PATH = PROJECT_ROOT / "assets" / "asl_chart.png"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

COMMON_WORDS = [
    "HELLO", "WORLD", "HELP", "PLEASE", "THANK", "YOU",
    "YES", "NO", "GOOD", "MORNING", "NIGHT", "FINE", "NAME",
    "WHAT", "HOW", "WHERE", "WHEN", "WHY", "DEAF", "HEARING",
    "FRIEND", "LOVE", "LEARN", "SIGN", "LANGUAGE"
]


def landmark_list(image: np.ndarray, landmarks: Any) -> list[list[int]]:
    height, width = image.shape[:2]
    return [
        [
            min(int(point.x * width), width - 1),
            min(int(point.y * height), height - 1),
        ]
        for point in landmarks.landmark
    ]


def preprocess_landmarks(points: list[list[int]]) -> list[float]:
    base_x, base_y = points[0]
    relative_points = [[x - base_x, y - base_y] for x, y in points]
    flattened = [coordinate for point in relative_points for coordinate in point]
    max_value = max(map(abs, flattened))
    return [coordinate / max_value for coordinate in flattened] if max_value else flattened


def draw_hand(frame: np.ndarray, points: list[list[int]]) -> None:
    for start, end in HAND_CONNECTIONS:
        cv.line(frame, tuple(points[start]), tuple(points[end]), (39, 174, 96), 2)
    for point in points:
        cv.circle(frame, tuple(point), 4, (42, 76, 232), -1)


class ASLRecognizer:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.50,
        )
        self.classifier = KeyPointClassifier(model_path=str(MODEL_PATH))
        with LABEL_PATH.open(encoding="utf-8-sig") as f:
            self.labels = [row[0] for row in csv.reader(f)]
        self.prediction_filter = TemporalPredictionFilter()
        self.composer = GestureTextComposer()
        self.sentence = ""

    def process(self, image: np.ndarray | None):
        if image is None:
            return None, "--", f"Sentence: {self.sentence}"

        # Gradio passes RGB images
        h, w = image.shape[:2]
        if w > 640:
            scale = 640.0 / w
            image = cv.resize(image, (640, int(h * scale)))

        annotated = image.copy()
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        predicted_character = None
        probabilities = None
        points: list[list[int]] | None = None
        hand_detected = results.multi_hand_landmarks is not None

        if hand_detected:
            hand_landmarks = results.multi_hand_landmarks[0]
            points = landmark_list(annotated, hand_landmarks)
            class_index = self.classifier(preprocess_landmarks(points))
            predicted_character = self.labels[class_index]
            probabilities = self.classifier.last_probabilities

        smoothed = self.prediction_filter.update(probabilities)
        stable_character = (
            self.labels[smoothed.label_index] if smoothed.is_stable else None
        )
        entry = self.composer.update(stable_character, hand_detected)

        if entry.committed_character:
            self.sentence += entry.committed_character

        display_char = stable_character or (
            f"{predicted_character}..." if predicted_character else "--"
        )
        status_text = (
            f"Sign: {display_char} ({int(smoothed.confidence * 100)}%)"
            if smoothed.is_stable
            else f"Sign: {display_char}"
        )

        if points:
            draw_hand(annotated, points)

        # Draw overlay bar
        cv.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (20, 28, 45), -1)
        cv.putText(
            annotated,
            f"Detected: {display_char} | {entry.message}",
            (15, 34),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
            cv.LINE_AA,
        )

        return annotated, status_text, self.sentence

    def clear(self):
        self.sentence = ""
        self.prediction_filter.reset()
        self.composer.reset()
        return "", "--"

    def backspace(self):
        self.sentence = self.sentence[:-1]
        return self.sentence

    def add_space(self):
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        return self.sentence


recognizer = ASLRecognizer()


def create_app():
    with gr.Blocks(title="ASL Sign to Text") as demo:
        gr.Markdown(
            """
            # 🤟 ASL Sign to Text Conversion
            Real-time fingerspelling recognition powered by MediaPipe and Deep Learning.
            Stream your webcam below to see signs detected with live text composition.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                webcam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Live Camera",
                )
                output_image = gr.Image(
                    label="Real-time Tracking & Prediction",
                    type="numpy",
                )

            with gr.Column(scale=2):
                if CHART_PATH.exists():
                    gr.Image(
                        value=str(CHART_PATH),
                        label="ASL Alphabet Reference Chart",
                        interactive=False,
                    )
                status_box = gr.Textbox(
                    label="Current Gesture Status",
                    value="Sign: --",
                    interactive=False,
                )
                sentence_box = gr.Textbox(
                    label="Formed Sentence / Message",
                    value="",
                    interactive=False,
                    lines=3,
                )

                with gr.Row():
                    btn_space = gr.Button("␣ Space", variant="secondary")
                    btn_backspace = gr.Button("⌫ Delete", variant="secondary")
                    btn_clear = gr.Button("🗑️ Clear", variant="stop")

        # Streaming callback: processes frame and updates tracking + sentence
        webcam_in.stream(
            fn=recognizer.process,
            inputs=[webcam_in],
            outputs=[output_image, status_box, sentence_box],
            stream_every=0.1,
            time_limit=300,
        )

        btn_clear.click(fn=recognizer.clear, outputs=[sentence_box, status_box])
        btn_backspace.click(fn=recognizer.backspace, outputs=[sentence_box])
        btn_space.click(fn=recognizer.add_space, outputs=[sentence_box])

    return demo


demo = create_app()

if __name__ == "__main__":
    demo.launch()
