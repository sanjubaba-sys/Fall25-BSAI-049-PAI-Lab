import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
train_path = "images/train"
model_path = "model/emotion_model.h5"
os.makedirs("model", exist_ok=True)
emotion_map = {d: i for i, d in enumerate(sorted(os.listdir(train_path)))}
data = []
labels = []

print("Loading training images...")
for emotion in emotion_map:
    folder = os.path.join(train_path, emotion)
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (48, 48))
        data.append(image)
        labels.append(emotion_map[emotion])

data = np.array(data).reshape(-1, 48, 48, 1) / 255.0
labels = to_categorical(labels)

print("Training data loaded:", data.shape)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(emotion_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(data, labels, epochs=10, batch_size=32)

print("Saving model...")
model.save(model_path)

print("Model saved at:", model_path)
