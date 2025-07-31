import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string

# Define all labels including SPACE and DELETE
labels = list(string.ascii_uppercase) + ['SPACE', 'DELETE']
label_map = {label: idx for idx, label in enumerate(labels)}

# Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels_numeric = np.asarray(data_dict['labels'])

# Print loaded label map
print("📌 Label Map:")
for label, idx in label_map.items():
    print(f"  {idx}: {label}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_numeric, test_size=0.2, stratify=labels_numeric, random_state=42
)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model and label map
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'label_map': label_map
    }, f)

print("💾 Model and label_map saved to 'model.p'")
