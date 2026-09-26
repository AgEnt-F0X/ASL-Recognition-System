"""Browser-based ASL fingerspelling app for Streamlit deployment."""

from __future__ import annotations

import csv
import json
import threading
import urllib.request
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

COMMON_WORDS = [
    "HELLO", "WORLD", "HELP", "PLEASE", "THANK", "YOU",
    "YES", "NO", "GOOD", "MORNING", "NIGHT", "FINE", "NAME",
    "WHAT", "HOW", "WHERE", "WHEN", "WHY", "DEAF", "HEARING",
    "FRIEND", "LOVE", "LEARN", "SIGN", "LANGUAGE"
]


@st.cache_data(ttl=1800)
def fetch_metered_turn(api_key: str, app_name: str = "") -> list[dict[str, Any]]:
    """Fetch ephemeral TURN credentials from Metered.ca API."""
    domain = f"{app_name}.metered.live" if app_name else "global.metered.live"
    url = f"https://{domain}/api/v1/turn/credentials?apiKey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Streamlit-ASL"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
    except Exception as exc:
        print(f"Error fetching Metered TURN credentials: {exc}")
    return []


@st.cache_data(ttl=1800)
def fetch_twilio_turn(account_sid: str, auth_token: str) -> list[dict[str, Any]]:
    """Fetch ephemeral TURN credentials from Twilio Network Traversal API."""
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as exc:
        print(f"Error fetching Twilio TURN credentials: {exc}")
    return []


def get_ice_servers() -> list[dict[str, Any]]:
    """Return configured ICE servers (STUN + optional TURN for cloud environments)."""
    if "ice_servers" in st.secrets:
        return list(st.secrets["ice_servers"])

    if "METERED_API_KEY" in st.secrets:
        servers = fetch_metered_turn(
            st.secrets["METERED_API_KEY"],
            st.secrets.get("METERED_APP_NAME", ""),
        )
        if servers:
            return servers

    if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
        servers = fetch_twilio_turn(
            st.secrets["TWILIO_ACCOUNT_SID"],
            st.secrets["TWILIO_AUTH_TOKEN"],
        )
        if servers:
            return servers

    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
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
            print(f"Live frame error: {exc}")
            return frame

    def process_static_image(self, bgr_image: np.ndarray) -> tuple[np.ndarray, str | None, float]:
        """Process a single image snapshot from the browser camera."""
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

    def apply_word(self, word: str) -> None:
        with self._lock:
            words = self._sentence.rstrip().split(" ")
            if words:
                words[-1] = word
            else:
                words = [word]
            self._sentence = " ".join(words) + " "

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
      .stButton button { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "recognition_session" not in st.session_state:
    st.session_state.recognition_session = RecognitionSession()
session: RecognitionSession = st.session_state.recognition_session

st.title("🤟 ASL Sign to Text")
st.caption("Fingerspelling recognition powered by MediaPipe & Deep Learning")

camera_column, reference_column = st.columns((2, 1), gap="large")

with camera_column:
    tab_snapshot, tab_live = st.tabs(["📸 Snapshot Camera (Recommended)", "🎥 Live Video Stream (WebRTC)"])

    with tab_snapshot:
        st.write("Hold your hand sign in the camera frame and click **Take photo**:")
        camera_snap = st.camera_input("Capture gesture", label_visibility="collapsed")
        if camera_snap is not None:
            raw_bytes = np.asarray(bytearray(camera_snap.read()), dtype=np.uint8)
            snap_img = cv.imdecode(raw_bytes, cv.IMREAD_COLOR)
            processed_img, recognized_char, conf = session.process_static_image(snap_img)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.image(
                    cv.cvtColor(processed_img, cv.COLOR_BGR2RGB),
                    caption="Analyzed Hand Landmarks",
                    use_column_width=True,
                )
            with res_col2:
                if recognized_char:
                    st.metric("Detected Sign", recognized_char)
                    st.metric("Confidence", f"{conf:.0%}")
                    if st.button(f"➕ Add '{recognized_char}' to Message", type="primary", use_container_width=True):
                        session.add_character(recognized_char)
                        st.rerun()
                else:
                    st.warning("⚠️ No hand detected. Hold your hand clearly in front of the camera with good lighting.")

    with tab_live:
        has_turn_secrets = any(
            k in st.secrets for k in ("METERED_API_KEY", "TWILIO_ACCOUNT_SID", "ice_servers")
        )

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
            st.success("🟢 Live stream connected")
        else:
            if not has_turn_secrets:
                st.info(
                    "💡 **Note for Cloud Users:** If the video stream resets or stays on 'START', "
                    "your network or cloud container blocks direct WebRTC UDP packets. "
                    "Use the **Snapshot Camera** tab, or add a free TURN relay key to Secrets."
                )
                with st.expander("🛠️ How to enable Live Video on Streamlit Cloud (2 minutes)"):
                    st.markdown(
                        """
                        Streamlit Community Cloud containers block incoming peer-to-peer UDP connections. 
                        WebRTC requires a TURN server to relay video over HTTPS ports.

                        **To enable live continuous streaming for free:**
                        1. Create a free account at [metered.ca/stun-turn](https://www.metered.ca/stun-turn) (50 GB free monthly, no card required).
                        2. In your Streamlit app, click **`⋮`** (bottom right) ➔ **Settings** ➔ **Secrets**.
                        3. Add:
                        ```toml
                        METERED_API_KEY = "your_api_key_from_metered"
                        ```
                        4. Save. Live video will now connect seamlessly through the TURN relay!
                        """
                    )

    # Word suggestions
    current_text = session.snapshot()["sentence"]
    words = current_text.split(" ") if current_text else []
    last_word = words[-1].upper() if words else ""
    suggestions = [w for w in COMMON_WORDS if w.startswith(last_word)][:4] if last_word else ["HELLO", "YES", "NO", "THANK"]

    st.write("**Suggestions:**")
    sugg_cols = st.columns(len(suggestions))
    for i, s_word in enumerate(suggestions):
        if sugg_cols[i].button(s_word, key=f"sugg_{s_word}_{i}", use_container_width=True):
            session.apply_word(s_word)
            st.rerun()

    # Readout & Message display
    state = session.snapshot()
    st.text_area("Formed Message", value=state["sentence"], height=90, disabled=True)

    c_clear, c_delete, c_space, c_speak = st.columns(4)
    if c_clear.button("🗑️ Clear", use_container_width=True):
        session.clear()
        st.rerun()
    if c_delete.button("⌫ Delete", use_container_width=True):
        session.delete_last()
        st.rerun()
    if c_space.button("␣ Space", use_container_width=True):
        session.add_space()
        st.rerun()
    if c_speak.button("🔊 Speak", type="primary", use_container_width=True):
        sentence_to_speak = session.snapshot()["sentence"].strip()
        if sentence_to_speak:
            speak_in_browser(sentence_to_speak)

with reference_column:
    st.image(str(CHART_PATH), caption="ASL alphabet reference", use_column_width=True)
