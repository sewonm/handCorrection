import cv2
import mediapipe as mp
import time
import math

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to calculate Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate angle between three points (e.g., wrist and two fingertips)
def calculate_angle(x1, y1, x2, y2, x3, y3):
    a = calculate_distance(x2, y2, x3, y3)
    b = calculate_distance(x1, y1, x3, y3)
    c = calculate_distance(x1, y1, x2, y2)
    # Using the cosine rule to calculate the angle
    try:
        angle = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
        return math.degrees(angle)
    except:
        return None

# Main loop for processing video feed
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the image for a mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(rgb_frame)

    # Define fingertip IDs and wrist ID
    fingertip_ids = [4, 8, 12, 16, 20]
    wrist_id = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Store fingertip and wrist coordinates
            points = {}
            for idx, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                if idx in fingertip_ids or idx == wrist_id:
                    points[idx] = (x, y)
                    # Draw circles on detected points
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)

            # Ensure all required points are detected
            if len(points) == 6:
                # Calculate distances between wrist and each fingertip
                for tip_id in fingertip_ids:
                    dist = calculate_distance(points[wrist_id][0], points[wrist_id][1], points[tip_id][0], points[tip_id][1])
                    cv2.putText(frame, f"Dist {tip_id}: {int(dist)}", (10, 30 + 20 * fingertip_ids.index(tip_id)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Calculate angles between wrist and fingertips (e.g., angle between thumb, wrist, and index finger)
                angle_thumb_index = calculate_angle(points[4][0], points[4][1], points[0][0], points[0][1], points[8][0], points[8][1])
                if angle_thumb_index:
                    cv2.putText(frame, f"Angle: {int(angle_thumb_index)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Hand Perspective Analysis", frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


