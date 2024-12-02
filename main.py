import cv2
import mediapipe as mp
import math

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)

# Calculate Euclidean distance
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Show distance between thumb and index finger
def show_distance_thumb_index(frame, hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    distance = calculate_distance(thumb_tip, index_tip)
    text = f"Distance: {distance:.2f}"
    cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    height, width, _ = frame.shape
    x1, y1 = int(thumb_tip.x * width), int(thumb_tip.y * height)
    x2, y2 = int(index_tip.x * width), int(index_tip.y * height)
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Recognize gestures
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    thumb_middle_distance = calculate_distance(thumb_tip, middle_tip)
    pinky_ring_distance = calculate_distance(pinky_tip, ring_tip)
    index_middle_distance = calculate_distance(index_tip, middle_tip)

    # Define gestures
    if thumb_index_distance < 0.05:  # OK Sign
        return "Gesture: OK Sign"
    elif thumb_index_distance > 0.2 and thumb_middle_distance > 0.2:
        return "Gesture: Open Hand"
    elif pinky_ring_distance < 0.05 and index_middle_distance > 0.15:  # Peace Sign
        return "Gesture: Peace Sign"
    elif thumb_index_distance < 0.1 and pinky_ring_distance > 0.15:  # Thumbs Up
        return "Gesture: Thumbs Up"
    elif thumb_index_distance > 0.15 and pinky_ring_distance < 0.05:  # Fist
        return "Gesture: Fist"
    else:
        return "Gesture: Unknown"

# Main loop
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show distance between thumb and index finger
            show_distance_thumb_index(frame, hand_landmarks)

            # Recognize and display gesture
            gesture = recognize_gesture(hand_landmarks.landmark)
            cv2.putText(frame, gesture, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
