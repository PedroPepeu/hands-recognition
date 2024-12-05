import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model('gesture_recognition_model.h5')

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Find the bounding box of the hand
            min_x = min([lm.x for lm in hand_landmarks.landmark])
            max_x = max([lm.x for lm in hand_landmarks.landmark])
            min_y = min([lm.y for lm in hand_landmarks.landmark])
            max_y = max([lm.y for lm in hand_landmarks.landmark])

            # Convert from normalized coordinates to pixel values
            h, w, _ = frame.shape
            min_x, max_x = int(min_x * w), int(max_x * w)
            min_y, max_y = int(min_y * h), int(max_y * h)

            # Draw a bounding box around the hand
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            # Optional: Draw landmarks (for debugging)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame with the bounding box
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
