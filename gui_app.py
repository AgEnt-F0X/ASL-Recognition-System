import customtkinter as ctk
from PIL import Image
import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools
import pyttsx3
import threading
from gesture_text import GestureTextComposer, TemporalPredictionFilter

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# --- Helper Functions (Copied/Adapted from app.py to avoid import execution) ---
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def draw_landmarks_on_white(image_size, landmark_point):
    # Create a white image
    white_img = np.ones(image_size, dtype=np.uint8) * 255
    
    if len(landmark_point) > 0:
        connections = [
            (2, 3), (3, 4),               # Thumb
            (5, 6), (6, 7), (7, 8),       # Index
            (9, 10), (10, 11), (11, 12),  # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20), # Little
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0) # Palm
        ]
        
        # Draw lines (Green for skeleton)
        for p1, p2 in connections:
             cv.line(white_img, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (0, 255, 0), 4)

        # Draw keypoints (Red circles)
        for index, landmark in enumerate(landmark_point):
            cv.circle(white_img, (landmark[0], landmark[1]), 5, (0, 0, 255), -1)

    return white_img

def draw_landmarks_on_image(image, landmark_point):
    """Draw hand skeletal connections and joints directly on the camera frame."""
    if len(landmark_point) > 0:
        connections = [
            (2, 3), (3, 4),               # Thumb
            (5, 6), (6, 7), (7, 8),       # Index
            (9, 10), (10, 11), (11, 12),  # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20), # Little
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0) # Palm
        ]
        for p1, p2 in connections:
             cv.line(image, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (0, 255, 0), 2)
        for index, landmark in enumerate(landmark_point):
            cv.circle(image, (landmark[0], landmark[1]), 4, (0, 0, 255), -1)
    return image

# --- Main Application Class ---
class ASLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language To Text Conversion")
        self.geometry("1400x820")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Configuration
        self.cap_device = 0
        self.cap_width = 640
        self.cap_height = 480
        self.min_detection_confidence = 0.65
        self.min_tracking_confidence = 0.5
        
        # State
        self.current_sentence = ""
        self.last_char = ""
        
        # Smoothing and deliberate letter entry prevent one noisy frame or a
        # held sign from producing incorrect or duplicate text.
        self.prediction_filter = TemporalPredictionFilter()
        self.text_composer = GestureTextComposer()
        
        # Suggestions Data
        self.common_words = [
            "HELLO", "HELP", "HERE", "HOME", "HOW", "HAPPY", 
            "YES", "YOU", "YOUR", "YEAR", 
            "NO", "NOT", "NOW", "NAME", "NICE", 
            "THANK", "THAT", "THIS", "THEY", "TIME",
            "PLEASE", "PEOPLE", "PLAY",
            "GOOD", "GREAT", "GO",
            "WHAT", "WHERE", "WHEN", "WHY", "WHO",
            "I", "IS", "IN", "IT",
            "MY", "ME", "MORE",
            "A", "AND", "ARE", "ABOUT", "ALL"
        ]
        self.suggestion_buttons = []
        
        # TTS Engine - initialized on demand per thread or globally
        # We will init locally in thread to fix "only works once" issue
        
        # Model Initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.keypoint_classifier = KeyPointClassifier()
        
        # Read labels
        with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        # Setup UI
        self._setup_ui()
        
        # Camera Setup
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Keyboard shortcuts for quick control
        self.bind("<space>", lambda event: self.add_space())
        self.bind("<BackSpace>", lambda event: self.backspace_text())
        self.bind("<Return>", lambda event: self.speak_text())
        self.bind("<Escape>", lambda event: self.clear_text())

        # Start Processing
        self.process_frame()

    def _setup_ui(self):
        # Grid Configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main Container
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0) # Title
        self.main_frame.grid_rowconfigure(1, weight=1) # Visuals
        self.main_frame.grid_rowconfigure(2, weight=0) # Text & Controls

        # 1. Header
        self.label_title = ctk.CTkLabel(self.main_frame, text="Sign Language To Text Conversion", font=("Roboto", 28, "bold"))
        self.label_title.grid(row=0, column=0, columnspan=3, pady=(10, 15))

        # 2. Visuals Area (Left: Camera, Center: Skeleton, Right: Chart)
        
        # Left: Camera Feed
        self.frame_camera = ctk.CTkFrame(self.main_frame)
        self.frame_camera.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.label_camera_title = ctk.CTkLabel(self.frame_camera, text="Live Feed (with Tracking)", font=("Roboto", 16))
        self.label_camera_title.pack(pady=5)
        self.label_camera = ctk.CTkLabel(self.frame_camera, text="")
        self.label_camera.pack(expand=True, fill="both", padx=5, pady=5)

        # Center: Skeleton Feed
        self.frame_skeleton = ctk.CTkFrame(self.main_frame)
        self.frame_skeleton.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.label_skeleton_title = ctk.CTkLabel(self.frame_skeleton, text="Hand Skeleton", font=("Roboto", 16))
        self.label_skeleton_title.pack(pady=5)
        self.label_skeleton = ctk.CTkLabel(self.frame_skeleton, text="") 
        self.label_skeleton.pack(expand=True, fill="both", padx=5, pady=5)

        # Right: Reference Chart
        self.frame_chart = ctk.CTkFrame(self.main_frame)
        self.frame_chart.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        self.label_chart_title = ctk.CTkLabel(self.frame_chart, text="Reference Chart", font=("Roboto", 16))
        self.label_chart_title.pack(pady=5)
        
        # Load Chart Image
        try:
            chart_img = Image.open("assets/asl_chart.png")
            chart_img.thumbnail((380, 380))
            self.chart_ctk_img = ctk.CTkImage(light_image=chart_img, dark_image=chart_img, size=chart_img.size)
            self.label_chart = ctk.CTkLabel(self.frame_chart, text="", image=self.chart_ctk_img)
            self.label_chart.pack(expand=True, pady=10)
        except Exception:
            self.label_chart = ctk.CTkLabel(self.frame_chart, text="Chart not found")
            self.label_chart.pack(expand=True)

        # 3. Controls & Text Area
        self.frame_controls = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frame_controls.grid(row=2, column=0, columnspan=3, padx=15, pady=15, sticky="ew")
        
        # Left side: Text Info & Progress
        self.frame_text = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_text.pack(side="left", fill="x", expand=True)

        self.label_char = ctk.CTkLabel(self.frame_text, text="Character: --", font=("Roboto", 20, "bold"))
        self.label_char.pack(anchor="w")

        self.label_entry_hint = ctk.CTkLabel(self.frame_text, text="Hold a clear sign to add a letter", font=("Roboto", 13))
        self.label_entry_hint.pack(anchor="w", pady=(0, 3))

        # Visual progress bar for holding the gesture
        self.prog_hold = ctk.CTkProgressBar(self.frame_text, width=280, height=12)
        self.prog_hold.pack(anchor="w", pady=(3, 8))
        self.prog_hold.set(0.0)
        
        self.label_sentence = ctk.CTkLabel(self.frame_text, text="Sentence: ", font=("Roboto", 20, "bold"), text_color="#3498DB")
        self.label_sentence.pack(anchor="w")

        # Middle: Suggestions
        self.frame_suggestions = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_suggestions.pack(side="left", padx=15)
        
        self.lbl_suggestions = ctk.CTkLabel(self.frame_suggestions, text="Words:", font=("Roboto", 15, "bold"), text_color="#E74C3C")
        self.lbl_suggestions.pack(side="left", padx=5)

        for i in range(4):
            btn = ctk.CTkButton(self.frame_suggestions, text="", width=75, height=35, command=lambda x=i: self.use_suggestion(x))
            btn.pack(side="left", padx=4)
            btn.pack_forget()
            self.suggestion_buttons.append(btn)
        
        self.update_suggestions() 

        # Right: Action Buttons
        self.frame_actions = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_actions.pack(side="right")

        self.btn_space = ctk.CTkButton(self.frame_actions, text="Space", width=75, height=38, command=self.add_space)
        self.btn_space.pack(side="left", padx=5)

        self.btn_backspace = ctk.CTkButton(self.frame_actions, text="⌫ Delete", width=85, height=38, command=self.backspace_text)
        self.btn_backspace.pack(side="left", padx=5)

        self.btn_clear = ctk.CTkButton(self.frame_actions, text="Clear", width=75, height=38, command=self.clear_text)
        self.btn_clear.pack(side="left", padx=5)

        self.btn_speak = ctk.CTkButton(self.frame_actions, text="🔊 Speak", width=85, height=38, command=self.speak_text)
        self.btn_speak.pack(side="left", padx=5)

    def process_frame(self):
        ret, image = self.cap.read()
        if not ret:
            self.after(10, self.process_frame)
            return

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        
        # Process Image with MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        skeleton_img = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        
        predicted_char = None
        probabilities = None
        conf = 0.0
        hand_detected = results.multi_hand_landmarks is not None
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, _handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Prediction
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                conf = getattr(self.keypoint_classifier, 'last_confidence', 1.0)
                probabilities = self.keypoint_classifier.last_probabilities
                predicted_char = self.keypoint_classifier_labels[hand_sign_id]
                
                # Draw skeleton and bounding box directly on live camera feed!
                draw_landmarks_on_image(debug_image, landmark_list)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
                cv.putText(debug_image, f"{predicted_char} ({int(conf * 100)}%)",
                           (brect[0], max(25, brect[1] - 8)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

                # Draw on separate white canvas
                skeleton_img = draw_landmarks_on_white((image.shape[0], image.shape[1], 3), landmark_list)
        
        # Average the complete probability vector over several frames. This is
        # much more reliable than voting on individual argmax labels.
        smoothed = self.prediction_filter.update(probabilities)
        smoothed_char = (
            self.keypoint_classifier_labels[smoothed.label_index]
            if smoothed.is_stable
            else None
        )
        entry = self.text_composer.update(smoothed_char, hand_detected)
        self.prog_hold.set(entry.progress)

        if entry.committed_character:
            self.add_to_sentence(entry.committed_character)

        if smoothed_char:
            self.label_char.configure(
                text=f"Character: {smoothed_char} ({int(smoothed.confidence * 100)}%)"
            )
        elif predicted_char:
            self.label_char.configure(text=f"Character: {predicted_char} - stabilizing")
        else:
            self.label_char.configure(text="Character: --")
        self.label_entry_hint.configure(text=entry.message)
        
        # Update Camera Feed
        img_cam = cv.resize(debug_image, (400, 300))
        img_cam = cv.cvtColor(img_cam, cv.COLOR_BGR2RGB)
        img_cam_pil = Image.fromarray(img_cam)
        cam_ctk_img = ctk.CTkImage(light_image=img_cam_pil, dark_image=img_cam_pil, size=(400, 300))
        self.label_camera.configure(image=cam_ctk_img)
        self.label_camera.image = cam_ctk_img

        # Update Skeleton Feed
        img_skel = cv.resize(skeleton_img, (400, 300))
        img_skel = cv.cvtColor(img_skel, cv.COLOR_BGR2RGB) 
        img_skel_pil = Image.fromarray(img_skel)
        skel_ctk_img = ctk.CTkImage(light_image=img_skel_pil, dark_image=img_skel_pil, size=(400, 300))
        self.label_skeleton.configure(image=skel_ctk_img)
        self.label_skeleton.image = skel_ctk_img

        self.after(10, self.process_frame)

    def add_to_sentence(self, char):
        self.current_sentence += char
        self.label_sentence.configure(text=f"Sentence: {self.current_sentence}")
        self.update_suggestions()

    def add_space(self):
        if self.current_sentence and not self.current_sentence.endswith(" "):
            self.current_sentence += " "
            self.label_sentence.configure(text=f"Sentence: {self.current_sentence}")
            self.update_suggestions()

    def backspace_text(self):
        if self.current_sentence:
            self.current_sentence = self.current_sentence[:-1]
            self.label_sentence.configure(text=f"Sentence: {self.current_sentence}")
            self.update_suggestions()

    def clear_text(self):
        self.current_sentence = ""
        self.label_sentence.configure(text="Sentence: ")
        self.prediction_filter.reset()
        self.text_composer.reset()
        self.prog_hold.set(0.0)
        self.label_entry_hint.configure(text="Hold a clear sign to add a letter")
        self.update_suggestions()

    def speak_text(self):
        text_to_speak = self.current_sentence.strip()
        if text_to_speak:
            def speak():
                try:
                    engine = pyttsx3.init()
                    engine.say(text_to_speak)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")
            
            threading.Thread(target=speak, daemon=True).start()
    
    def update_suggestions(self):
        words = self.current_sentence.split(" ")
        current_fragment = words[-1].upper() if words else ""
        
        matches = []
        if current_fragment:
            matches = [w for w in self.common_words if w.startswith(current_fragment)]
        else:
            matches = ["HELLO", "YES", "NO", "THANK"]
            
        display_suggestions = matches[:4]
        
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(display_suggestions):
                btn.configure(text=display_suggestions[i])
                btn.pack(side="left", padx=4)
            else:
                btn.pack_forget()

    def use_suggestion(self, index):
        word = self.suggestion_buttons[index].cget("text")
        words = self.current_sentence.rstrip().split(" ")
        if words:
             words[-1] = word
        else:
             words = [word]
             
        self.current_sentence = " ".join(words) + " "
        self.label_sentence.configure(text=f"Sentence: {self.current_sentence}")
        self.update_suggestions()

    def on_closing(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv.destroyAllWindows()
        self.destroy()

if __name__ == "__main__":
    app = ASLApp()
    app.mainloop()
