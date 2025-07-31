import os
import cv2
import string

# Directory to save collected data
DATA_DIR = './data'

# Create main data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define gesture labels: A-Z, SPACE, DELETE
labels = list(string.ascii_uppercase) + ['SPACE', 'DELETE']
dataset_size = 100  # Number of images per gesture

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

# Loop through each gesture
for label in labels:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'🟢 Collecting data for: {label}')
    
    # Wait for user to press 'q' to begin
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            continue

        cv2.putText(frame, f'Get ready for: {label} (Press Q)', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Start image capture
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        print(f'Saved {counter+1}/{dataset_size} images for {label}')
        counter += 1

print("✅ Data collection complete!")
cap.release()
cv2.destroyAllWindows()
