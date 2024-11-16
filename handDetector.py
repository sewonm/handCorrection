import cv2
import mediapipe as mp
import time

# Initialize webcam and Mediapipe modules
camera = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
drawer = mp.solutions.drawing_utils

# Variables to track frame rate
previous_time = 0

# Main loop for processing video feed
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        print("Failed to capture image.")
        break

    # Flip the image for a mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image for hand landmarks
    detection_results = hand_detector.process(rgb_frame)

    # If landmarks are detected, process each hand
    if detection_results.multi_hand_landmarks:
        for landmarks in detection_results.multi_hand_landmarks:
            frame_height, frame_width, _ = frame.shape

            # Extract specific landmarks for perspective analysis (e.g., fingertips)
            fingertip_ids = [4, 8, 12, 16, 20]
            for point_id, landmark in enumerate(landmarks.landmark):
                # Calculate pixel coordinates
                x_pixel = int(landmark.x * frame_width)
                y_pixel = int(landmark.y * frame_height)

                # Highlight fingertip points
                if point_id in fingertip_ids:
                    cv2.circle(frame, (x_pixel, y_pixel), 10, (0, 255, 0), cv2.FILLED)

            # Draw the hand landmarks and connections
            drawer.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculate and display FPS
    current_time = time.time()
    fps = int(1 / (current_time - previous_time)) if (current_time - previous_time) > 0 else 0
    previous_time = current_time
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed video feed
    cv2.imshow("Perspective-Corrected Hand Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()


# /Users/sewonmyung/myenv/bin/python /Users/sewonmyung/programming/handDetector.py
