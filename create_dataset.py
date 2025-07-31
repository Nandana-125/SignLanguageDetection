import os
import pickle
import string
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define data directory
DATA_DIR = './data'

data = []
labels = []

# Create label_map: {'A': 0, 'B': 1, ..., 'Z': 25}
label_map = {letter: idx for idx, letter in enumerate(string.ascii_uppercase)}

print("🔍 Starting data preprocessing...\n")

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # skip any non-directories like .DS_Store

    print(f"📁 Processing class: {dir_}")

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            x_ = []
            y_ = []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(label_map[dir_])

print(f"\n✅ Processed {len(data)} samples.")

# Save data with label_map
with open('data.pickle', 'wb') as f:
    pickle.dump({
        'data': data,
        'labels': labels,
        'label_map': label_map
    }, f)

print("💾 Saved to data.pickle successfully.")
