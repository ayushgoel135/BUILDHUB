import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pandas as pd
import time
from enum import Enum, auto
import pygame
import os
from twilio.rest import Client
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import tempfile
from PIL import Image

# Constants
DEFAULT_FRAME_SIZE = (640, 480)
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MAX_CONSECUTIVE_ERRORS = 5
FPS = 30


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()


class ExerciseStage(Enum):
    UP = auto()
    DOWN = auto()
    REST = auto()


@dataclass
class ExerciseConfig:
    joints: Tuple[str, str, str]
    angle_range: Tuple[float, float]
    instructions: str
    completion_threshold: float = 0.9
    rest_threshold: float = 1.1


# Initialize pygame for audio
pygame.mixer.init()

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Audio Configuration
class AudioManager:
    def __init__(self):
        self.volume = 0.7
        self.enabled = True
        self.sounds = {
            "correct": self._load_sound("good.mp3"),
            "incorrect": self._load_sound("good.mp3"),
            
            "complete": self._load_sound("man-says-amazing-184036.mp3")
        }

    def _load_sound(self, path: str) -> Optional[pygame.mixer.Sound]:
        if not os.path.exists(path):
            st.warning(f"Audio file not found: {path}")
            return None
        try:
            return pygame.mixer.Sound(path)
        except Exception as e:
            st.warning(f"Couldn't load audio {path}: {str(e)}")
            return None

    def play(self, sound_name: str):
        if not self.enabled:
            return

        sound = self.sounds.get(sound_name)
        if sound:
            sound.set_volume(self.volume)
            sound.play()


audio_manager = AudioManager()


# Pose Detection Utilities
class PoseAnalyzer:
    @staticmethod
    def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    @staticmethod
    def get_landmark_points(landmarks, side: Side, joints: Tuple[str, str, str]) -> Tuple[float, List[float]]:
        prefix = "LEFT_" if side == Side.LEFT else "RIGHT_"
        points = [
            [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[0]}").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[0]}").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[1]}").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[1]}").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[2]}").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[2]}").value].y]
        ]
        angle = PoseAnalyzer.calculate_angle(*points)
        return angle, points[1]


# Feedback System
class FeedbackSystem:
    def __init__(self):
        self.consecutive_errors = 0
        self.notification_sent = False

    def analyze_form(self, exercise: str, left_angle: float, right_angle: float,
                     config: ExerciseConfig) -> Tuple[List[str], bool]:
        feedback = []
        critical_error = False

        # Check balance between sides
        if abs(left_angle - right_angle) > 15:
            feedback.append("⚠️ Keep both sides balanced! Significant asymmetry detected.")
            critical_error = True

        # Check range of motion
        if left_angle < config.angle_range[0] or right_angle < config.angle_range[0]:
            feedback.append(f"⬇️ Increase range! Should be {config.angle_range[0]}-{config.angle_range[1]}°")
            critical_error = True
        elif left_angle > config.angle_range[1] or right_angle > config.angle_range[1]:
            feedback.append(f"⬆️ Reduce range! Should be {config.angle_range[0]}-{config.angle_range[1]}°")
            critical_error = True

        # Exercise-specific checks
        if exercise == "Bicep Curl" and (left_angle > 160 or right_angle > 160):
            feedback.append("💪 Keep elbows close to your body!")
            critical_error = True
        elif exercise == "Squat" and (left_angle < 70 or right_angle < 70):
            feedback.append("🦵 Go deeper! Knees should bend more.")
            critical_error = True

        # Update error tracking
        if critical_error:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 2:
                feedback.append("🔴 Multiple form issues detected!")
        else:
            self.consecutive_errors = 0

        return feedback if feedback else ["👍 Perfect form! Keep it up!"], critical_error


# Notification System
class NotificationManager:
    def __init__(self):
        self.client = None
        self.enabled = False
        self.phone_number = ""
        self.threshold = 3

    def initialize(self, account_sid: str, auth_token: str):
        try:
            self.client = Client(account_sid, auth_token)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Twilio client: {str(e)}")
            return False

    def send_notification(self, message: str):
        if not self.enabled or not self.client or not self.phone_number:
            return False

        try:
            self.client.messages.create(
                body=message,
                from_=st.secrets["TWILIO_PHONE_NUMBER"],
                to=self.phone_number
            )
            return True
        except Exception as e:
            st.error(f"Failed to send notification: {str(e)}")
            return False


# Exercise Configuration
EXERCISE_CONFIGS = {
    "Bicep Curl": ExerciseConfig(
        joints=("SHOULDER", "ELBOW", "WRIST"),
        angle_range=(30, 160),
        instructions="Keep elbows close to your body and control the movement.",
        completion_threshold=0.85
    ),
    "Squat": ExerciseConfig(
        joints=("HIP", "KNEE", "ANKLE"),
        angle_range=(70, 160),
        instructions="Keep knees aligned with toes and back straight.",
        completion_threshold=0.8
    ),
    "Shoulder Press": ExerciseConfig(
        joints=("ELBOW", "SHOULDER", "HIP"),
        angle_range=(60, 140),
        instructions="Don't lock elbows at the top, maintain control.",
        completion_threshold=0.9
    ),
    "Push-ups": ExerciseConfig(
        joints=("SHOULDER", "ELBOW", "WRIST"),
        angle_range=(70, 150),
        instructions="Keep body straight, don't sag at the hips.",
        completion_threshold=0.85
    ),
    "Lunges": ExerciseConfig(
        joints=("HIP", "KNEE", "ANKLE"),
        angle_range=(80, 170),
        instructions="Front knee should be above ankle, back knee pointing down.",
        completion_threshold=0.8
    )
}


# Streamlit UI Setup
def setup_ui():
    st.set_page_config(page_title="AI Gym Trainer Pro", layout="wide", page_icon="🏋️")
    st.title("🏋️ AI Gym Trainer Pro")

    # Sidebar Configuration
    with st.sidebar:
        st.header("Exercise Settings")
        selected_exercise = st.selectbox(
            "Choose Exercise",
            list(EXERCISE_CONFIGS.keys()),
            index=0
        )

        st.header("Workout Parameters")
        target_reps = st.number_input("Target Repetitions", 1, 100, 10)
        rest_time = st.number_input("Rest Between Sets (seconds)", 0, 300, 60)

        st.header("Feedback Settings")
        show_landmarks = st.checkbox("Show Pose Landmarks", True)
        audio_manager.enabled = st.checkbox("Enable Audio Feedback", True)
        audio_manager.volume = st.slider("Volume", 0.0, 1.0, 0.7)

        st.header("Notification Settings")
        notification_manager = NotificationManager()
        notification_manager.enabled = st.checkbox("Enable WhatsApp Notifications", False)
        if notification_manager.enabled:
            phone_number = st.text_input("Your WhatsApp Number", "+917011722230")
            if phone_number and not phone_number.startswith("whatsapp:"):
                phone_number = f"whatsapp:{phone_number}"
            notification_manager.phone_number = phone_number
            notification_manager.threshold = st.slider("Notification Threshold", 1, 5, 3)
            if st.button("Test Notification"):
                if notification_manager.send_notification("Test notification from AI Gym Trainer"):
                    st.success("Test notification sent!")

    return selected_exercise, target_reps, rest_time, show_landmarks, notification_manager


# Main Application
class AIGymTrainer:
    def __init__(self):
        self.cap = None
        self.exercise_data = {
            "Reps": [], "Correct": [], "Incorrect": [],
            "Left Angle": [], "Right Angle": [], "Time": []
        }
        self.start_time = None
        self.stage = ExerciseStage.REST
        self.counter = 0
        self.incorrect_counter = 0

    def start_session(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Could not open webcam")
            return False
        return True

    def end_session(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        pygame.mixer.quit()

    def process_frame(self, frame, exercise: str, show_landmarks: bool) -> Tuple[np.ndarray, List[str]]:
        config = EXERCISE_CONFIGS.get(exercise)
        feedback = []

        try:
            # Convert and process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = mp_pose.Pose(
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            ).process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                # Analyze pose
                left_angle, left_center = PoseAnalyzer.get_landmark_points(
                    results.pose_landmarks.landmark, Side.LEFT, config.joints
                )
                right_angle, right_center = PoseAnalyzer.get_landmark_points(
                    results.pose_landmarks.landmark, Side.RIGHT, config.joints
                )

                # Get feedback
                feedback, critical = FeedbackSystem().analyze_form(
                    exercise, left_angle, right_angle, config
                )

                # Check for rep completion
                avg_angle = (left_angle + right_angle) / 2
                if self._check_rep_completion(avg_angle, config):
                    self._handle_rep_completion(critical)

                # Draw landmarks and angles
                image = self._annotate_image(
                    image, results, show_landmarks,
                    left_angle, right_angle, left_center, right_center
                )

            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), feedback

        except Exception as e:
            st.warning(f"Frame processing error: {str(e)}")
            return frame, ["⚠️ System error - continuing..."]

    def _check_rep_completion(self, avg_angle: float, config: ExerciseConfig) -> bool:
        if avg_angle > config.angle_range[1] * config.completion_threshold:
            self.stage = ExerciseStage.DOWN
        elif (avg_angle < config.angle_range[0] * config.rest_threshold and
              self.stage == ExerciseStage.DOWN):
            return True
        return False

    def _handle_rep_completion(self, critical: bool):
        self.stage = ExerciseStage.UP
        self.counter += 1
        if not critical:
            audio_manager.play("correct")
        else:
            self.incorrect_counter += 1
            audio_manager.play("incorrect")

    def _annotate_image(self, image, results, show_landmarks,
                        left_angle, right_angle, left_center, right_center):
        # Draw angles
        cv2.putText(image, f'L: {int(left_angle)}°',
                    tuple(np.multiply(left_center, DEFAULT_FRAME_SIZE).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'R: {int(right_angle)}°',
                    tuple(np.multiply(right_center, DEFAULT_FRAME_SIZE).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw landmarks if enabled
        if show_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        return image

    def get_stats(self) -> Dict:
        return {
            "reps": self.counter,
            "incorrect": self.incorrect_counter,
            "accuracy": (self.counter - self.incorrect_counter) / self.counter * 100 if self.counter > 0 else 0,
            "elapsed": time.time() - self.start_time if self.start_time else 0
        }


# Main Execution
def main():
    # Setup UI and get settings
    selected_exercise, target_reps, rest_time, show_landmarks, notification_manager = setup_ui()

    # Initialize application
    app = AIGymTrainer()
    feedback_system = FeedbackSystem()

    # Create layout
    video_placeholder = st.empty()
    feedback_placeholder = st.empty()
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Start/Stop controls
    col1, col2 = st.columns(2)
    start_button = col1.button("Start Session")
    stop_button = col2.button("Stop Session")

    if start_button and app.start_session():
        app.start_time = time.time()
        last_notification_time = 0

        while app.cap.isOpened() and not stop_button and app.counter < target_reps:
            start_time = time.time()

            # Capture frame
            ret, frame = app.cap.read()
            if not ret:
                st.warning("Frame capture error")
                break

            # Process frame
            frame = cv2.resize(frame, DEFAULT_FRAME_SIZE)
            processed_frame, feedback = app.process_frame(frame, selected_exercise, show_landmarks)

            # Update UI
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            feedback_placeholder.markdown("### Feedback\n" + "\n".join(f"• {msg}" for msg in feedback))

            # Update stats
            stats = app.get_stats()
            stats_placeholder.markdown(f"""
                ### Session Stats
                - **Exercise**: {selected_exercise}
                - **Reps**: {stats['reps']}/{target_reps}
                - **Incorrect**: {stats['incorrect']}
                - **Accuracy**: {stats['accuracy']:.1f}%
                - **Elapsed**: {int(stats['elapsed'])}s
            """)

            # Check for notifications
            if (notification_manager.enabled and
                    feedback_system.consecutive_errors >= notification_manager.threshold and
                    time.time() - last_notification_time > 300):  # 5 minute cooldown

                message = (f"🚨 AI Gym Trainer Alert!\n"
                           f"Multiple incorrect {selected_exercise} attempts detected!\n"
                           f"Current stats: {stats['reps']} reps, {stats['incorrect']} incorrect")

                if notification_manager.send_notification(message):
                    last_notification_time = time.time()
                    feedback.append("📲 Notification sent about form issues")

            # Control frame rate
            elapsed = time.time() - start_time
            time.sleep(max(0, 1 / FPS - elapsed))

        # Session complete
        app.end_session()
        if app.counter >= target_reps:
            st.balloons()
            st.success(f"🎉 Completed {target_reps} reps of {selected_exercise}!")
            audio_manager.play("complete")

            # Show analytics
            st.subheader("Session Analytics")
            df = pd.DataFrame(app.exercise_data)
            st.line_chart(df[["Left Angle", "Right Angle"]])
            st.bar_chart(df[["Correct", "Incorrect"]])

            # Show summary
            st.write(f"""
                ### Summary
                - Total Time: {int(stats['elapsed'])} seconds
                - Average Rep Time: {stats['elapsed'] / app.counter:.1f}s per rep
                - Accuracy: {stats['accuracy']:.1f}%
            """)


if __name__ == "__main__":
    main()