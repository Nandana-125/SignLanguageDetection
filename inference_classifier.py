import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load model and label map
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
label_map = model_dict['label_map']
labels_dict = {v: k for k, v in label_map.items()}  # reverse map: 0 → 'A'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

sentence = ""
last_prediction = ""
last_time = time.time()
prediction_delay = 1.0  # seconds before accepting a new character

print("✋ Ready — show gestures. ESC to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        x_ = []
        y_ = []
        data_aux = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])[0]
                    predicted_char = labels_dict[int(prediction)]

                    # Draw box
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_char, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    # Add character to sentence if cooldown passed and it's a new letter
                    current_time = time.time()
                    if predicted_char != last_prediction and (current_time - last_time) > prediction_delay:
                        if predicted_char == "SPACE":
                            sentence += " "
                        elif predicted_char == "DELETE":
                            sentence = sentence[:-1]
                        else:
                            sentence += predicted_char
                        last_prediction = predicted_char
                        last_time = current_time

        # Show sentence
        cv2.rectangle(frame, (0, 0), (W, 60), (255, 255, 255), -1)
        cv2.putText(frame, f"Text: {sentence}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Sign Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

cap.release()
cv2.destroyAllWindows()
