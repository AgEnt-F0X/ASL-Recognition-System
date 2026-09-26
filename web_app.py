"""Browser-based ASL fingerspelling app for Streamlit deployment."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

import av
import cv2 as cv
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from gesture_text import GestureTextComposer, TemporalPredictionFilter
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


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


def get_ice_servers() -> list[dict[str, Any]]:
    """Return STUN and TURN configuration with open relay fallback for cloud NAT traversal."""
    if "ice_servers" in st.secrets:
        return st.secrets["ice_servers"]
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:openrelay.metered.ca:80"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
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


class RecognitionSession:
    """Owns inference state without calling Streamlit from WebRTC threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.50,
        )
        self._classifier = KeyPointClassifier(model_path=str(MODEL_PATH))
        with LABEL_PATH.open(encoding="utf-8-sig") as labels_file:
            self._labels = [row[0] for row in csv.reader(labels_file)]
        self._prediction_filter = TemporalPredictionFilter()
        self._composer = GestureTextComposer()
        self._sentence = ""
        self._current_character = "--"
        self._confidence = 0.0
        self._progress = 0.0
        self._hint = "Show one clear hand sign"

    def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            image = cv.flip(image, 1)

            # Cap frame width to 640px to conserve bandwidth and CPU
            h, w = image.shape[:2]
            if w > 640:
                scale = 640.0 / w
                image = cv.resize(image, (640, int(h * scale)))

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self._hands.process(image_rgb)
            image_rgb.flags.writeable = True

            predicted_character = None
            probabilities = None
            points: list[list[int]] | None = None
            hand_detected = results.multi_hand_landmarks is not None
            if hand_detected:
                hand_landmarks = results.multi_hand_landmarks[0]
                points = landmark_list(image, hand_landmarks)
                class_index = self._classifier(preprocess_landmarks(points))
                predicted_character = self._labels[class_index]
                probabilities = self._classifier.last_probabilities

            with self._lock:
                smoothed = self._prediction_filter.update(probabilities)
                stable_character = (
                    self._labels[smoothed.label_index] if smoothed.is_stable else None
                )
                entry = self._composer.update(stable_character, hand_detected)
                if entry.committed_character:
                    self._sentence += entry.committed_character

                self._current_character = stable_character or (
                    f"{predicted_character}..." if predicted_character else "--"
                )
                self._confidence = smoothed.confidence if smoothed.is_stable else 0.0
                self._progress = entry.progress
                self._hint = entry.message
                sentence = self._sentence

            if points:
                draw_hand(image, points)
            self._draw_overlay(image, sentence)
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as exc:
            # Prevent WebRTC crash on frame error
            print(f"Frame processing error: {exc}")
            return frame

    def process_static_image(self, bgr_image: np.ndarray) -> tuple[np.ndarray, str | None, float]:
        """Process a single image snapshot from browser camera or upload."""
        image = bgr_image.copy()
        h, w = image.shape[:2]
        if w > 640:
            scale = 640.0 / w
            image = cv.resize(image, (640, int(h * scale)))

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            points = landmark_list(image, hand_landmarks)
            class_index = self._classifier(preprocess_landmarks(points))
            char = self._labels[class_index]
            conf = self._classifier.last_confidence
            draw_hand(image, points)
            return image, char, conf
        return image, None, 0.0

    def add_character(self, char: str) -> None:
        with self._lock:
            self._sentence += char

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sentence": self._sentence,
                "character": self._current_character,
                "confidence": self._confidence,
                "progress": self._progress,
                "hint": self._hint,
            }

    def clear(self) -> None:
        with self._lock:
            self._sentence = ""
            self._prediction_filter.reset()
            self._composer.reset()
            self._current_character = "--"
            self._confidence = 0.0
            self._progress = 0.0
            self._hint = "Show one clear hand sign"

    def delete_last(self) -> None:
        with self._lock:
            self._sentence = self._sentence[:-1]

    def add_space(self) -> None:
        with self._lock:
            if self._sentence and not self._sentence.endswith(" "):
                self._sentence += " "

    def _draw_overlay(self, image: np.ndarray, sentence: str) -> None:
        height, width = image.shape[:2]
        cv.rectangle(image, (0, 0), (width, 88), (18, 27, 45), -1)
        cv.putText(
            image,
            f"Letter: {self._current_character}",
            (18, 34),
            cv.FONT_HERSHEY_SIMPLEX,
            0.85,
            (243, 244, 246),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            f"Text: {sentence[-28:] or '-'}",
            (18, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.72,
            (139, 221, 255),
            2,
            cv.LINE_AA,
        )


def speak_in_browser(text: str) -> None:
    payload = json.dumps(text)
    st.components.v1.html(
        f"<script>speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance({payload}));</script>",
        height=0,
    )


st.set_page_config(page_title="ASL Sign to Text", page_icon="🤟", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 1280px; padding-top: 1.5rem; }
      [data-testid="stMetric"] { border-left: 3px solid #23a6d5; padding-left: 0.8rem; }
      [data-testid="stImage"] img { border: 1px solid #d7dde5; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "recognition_session" not in st.session_state:
    st.session_state.recognition_session = RecognitionSession()
session: RecognitionSession = st.session_state.recognition_session

st.title("🤟 ASL Sign to Text")
st.caption("Real-time fingerspelling recognition")

camera_column, reference_column = st.columns((2, 1), gap="large")

with camera_column:
    mode_tab_live, mode_tab_snapshot = st.tabs(["🎥 Live Stream", "📸 Snapshot Camera"])

    with mode_tab_live:
        webrtc_ctx = webrtc_streamer(
            key="asl-sign-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=session.process_frame,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 20, "max": 30},
                },
                "audio": False,
            },
            rtc_configuration={"iceServers": get_ice_servers()},
            async_processing=True,
        )
        if webrtc_ctx and webrtc_ctx.state.playing:
            st.success("🟢 Camera stream connected & active")
        else:
            st.info("Click **START** above to begin live video. If your network blocks live WebRTC, switch to the **Snapshot Camera** tab!")

    with mode_tab_snapshot:
        st.write("Take a snapshot of your hand gesture:")
        camera_snap = st.camera_input("Capture gesture", label_visibility="collapsed")
        if camera_snap is not None:
            raw_bytes = np.asarray(bytearray(camera_snap.read()), dtype=np.uint8)
            snap_img = cv.imdecode(raw_bytes, cv.IMREAD_COLOR)
            processed_img, recognized_char, conf = session.process_static_image(snap_img)
            
            snap_col1, snap_col2 = st.columns(2)
            with snap_col1:
                st.image(
                    cv.cvtColor(processed_img, cv.COLOR_BGR2RGB),
                    caption="Analyzed Hand",
                    use_column_width=True,
                )
            with snap_col2:
                if recognized_char:
                    st.metric("Detected Letter", recognized_char)
                    st.metric("Confidence", f"{conf:.0%}")
                    if st.button(f"Add '{recognized_char}' to Message", use_container_width=True):
                        session.add_character(recognized_char)
                        st.rerun()
                else:
                    st.warning("No hand detected. Please hold your hand clearly in front of the camera.")

with reference_column:
    st.image(str(CHART_PATH), caption="ASL alphabet reference", use_column_width=True)


@st.fragment(run_every=0.4)
def live_readout() -> None:
    state = session.snapshot()
    letter_column, confidence_column = st.columns(2)
    letter_column.metric("Detected letter", state["character"])
    confidence_column.metric("Stable confidence", f"{state['confidence']:.0%}")
    st.progress(state["progress"], text=state["hint"])
    st.text_area("Message", value=state["sentence"], height=100, disabled=True)


live_readout()

clear_column, delete_column, space_column, speak_column = st.columns(4)
if clear_column.button("Clear", use_container_width=True):
    session.clear()
    st.rerun()
if delete_column.button("Delete", use_container_width=True):
    session.delete_last()
    st.rerun()
if space_column.button("Space", use_container_width=True):
    session.add_space()
    st.rerun()
if speak_column.button("Speak", use_container_width=True):
    sentence = session.snapshot()["sentence"].strip()
    if sentence:
        speak_in_browser(sentence)
