import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('cnn_trained_csv_28x28.h5')

# Create index-to-letter mapping (0-25 for A-Z)
index_to_letter = {i: chr(65 + i) for i in range(26)}

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Queue to hold last 10 predictions for smoothing
prediction_history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            hand_region = frame[y_min:y_max, x_min:x_max]

            gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            resized_hand = cv2.resize(gray_hand, (28, 28))
            normalized_hand = resized_hand / 255.0
            input_data = normalized_hand.reshape(1, 28, 28, 1)

            predictions = model.predict(input_data)
            predicted_class = np.argmax(predictions)

            # Append prediction to history and get most common prediction
            prediction_history.append(predicted_class)
            most_common_pred = max(set(prediction_history), key=prediction_history.count)
            predicted_letter = index_to_letter.get(most_common_pred, '')

            cv2.putText(frame, f'Prediction: {predicted_letter}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(frame, 'No hand detected', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Clear history if no hand detected to avoid stale predictions
        prediction_history.clear()

    cv2.putText(frame, "Press q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Tracking and Sign Prediction', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

