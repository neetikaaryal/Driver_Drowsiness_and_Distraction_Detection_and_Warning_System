"""
# python3.11
Drowsiness and Yawn Detection using Mediapipe Face Mesh

This script uses Mediapipe to detect facial landmarks and then computes:
  - The Eye Aspect Ratio (EAR) to determine if the eyes are closed (drowsiness)
  - The vertical distance between the lips to detect yawning

When drowsiness or yawning is detected, an alarm (printed message) is triggered
in a separate thread.
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import time

# -------------------------------
# Global parameters and thresholds
# -------------------------------
EYE_AR_THRESH = 0.15          # EAR threshold to indicate closed eyes
EYE_AR_CONSEC_FRAMES = 30    # Number of consecutive frames the eye must be below the threshold to trigger drowsiness alert
YAWN_THRESH = 70             # Lip distance (in pixels) threshold to trigger yawn alert

# Global flags and counters for alarms
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# -------------------------------
# Helper Functions
# -------------------------------
def alarm(msg):
    """
    Alarm function that prints an alert message.
    Runs in a separate thread so that it does not block the video stream.
    """
    global alarm_status, alarm_status2, saying

    # Continue sounding the alarm while either flag is True.
    while alarm_status or alarm_status2:
        print("ALARM:", msg)
        time.sleep(1)  # pause for a second between alerts


def eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio (EAR) given 6 eye landmarks.
    The eye landmarks should be provided in the following order:
      p1, p2, p3, p4, p5, p6
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def process_frame(frame, face_mesh, draw=True):
    """
    Process a single video frame to detect faces and compute:
      - Eye Aspect Ratio (EAR) for drowsiness detection
      - Lip distance for yawn detection

    The function annotates the frame with landmarks and alert text.

    Parameters:
      frame      : The input video frame (BGR format).
      face_mesh  : An initialized mediapipe FaceMesh object.
      draw       : If True, draw landmarks on the frame.

    Returns:
      The annotated frame.
    """
    global COUNTER, alarm_status, alarm_status2, saying

    height, width = frame.shape[:2]
    # Convert the frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # If faces are detected, process each one (we limit to one face here)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * width), int(lm.y * height)
                landmarks.append((x, y))
            landmarks = np.array(landmarks)

            # -----------------------------------------------------
            # Define eye indices for Mediapipe (6 landmarks per eye)
            # These indices are commonly used in drowsiness detection:
            #   Right eye: [33, 160, 158, 133, 153, 144]
            #   Left eye:  [362, 385, 387, 263, 373, 380]
            # -----------------------------------------------------
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye_indices  = [362, 385, 387, 263, 373, 380]

            # Extract the eye landmark coordinates
            right_eye = landmarks[right_eye_indices]
            left_eye  = landmarks[left_eye_indices]

            # Compute EAR for each eye and take the average
            right_ear = eye_aspect_ratio(right_eye)
            left_ear  = eye_aspect_ratio(left_eye)
            ear = (right_ear + left_ear) / 2.0

            # -----------------------------------------------------
            # Yawn detection: Compute lip distance.
            # For simplicity, we use two landmarks:
            #   Top lip: landmark index 13
            #   Bottom lip: landmark index 14
            # These indices are often used for mouth open detection.
            # -----------------------------------------------------
            top_lip = landmarks[13]
            bottom_lip = landmarks[14]
            lip_distance = abs(top_lip[1] - bottom_lip[1])

            # Optionally, draw the eye and lip landmarks on the frame
            if draw:
                for idx in right_eye_indices:
                    cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 255, 0), -1)
                for idx in left_eye_indices:
                    cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 255, 0), -1)
                cv2.circle(frame, tuple(top_lip), 2, (255, 0, 0), -1)
                cv2.circle(frame, tuple(bottom_lip), 2, (255, 0, 0), -1)

            # -------------------------------
            # Drowsiness Detection (using EAR)
            # -------------------------------
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=("Wake up, sir!",))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            # -------------------------------
            # Yawn Detection (using Lip Distance)
            # -------------------------------
            if lip_distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=("Take some fresh air, sir!",))
                    t.daemon = True
                    t.start()
            else:
                alarm_status2 = False

            # Display computed metrics on the frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Lip Distance: {lip_distance}", (width - 180, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Process only the first detected face for now
            break

    return frame

# -------------------------------
# Main function: Video Stream Loop
# -------------------------------
def main():
    """
    Main function to initialize Mediapipe Face Mesh and start the video capture loop.
    Processes each frame for drowsiness and yawn detection.
    """
    # Initialize Mediapipe Face Mesh with desired parameters
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    # Start video capture (default webcam)
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)  # Allow the camera to warm up

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Process the current frame
        frame = process_frame(frame, face_mesh, draw=True)

        # Display the output frame
        cv2.imshow("Drowsiness and Yawn Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit if 'q' is pressed
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Entry point of the script
# -------------------------------
if __name__ == "__main__":
    main()
