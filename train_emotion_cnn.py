# train emotion cnn model
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

MODEL_PATH = "emotion_cnn.h5"

# 1. Check if model exists
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
    print("Model loaded from disk.")
    model.summary()
    exit()

# 2. Load dataset from disk

img_height = 48
img_width = 48

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data/train',
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    'data/val',
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

print("Class mapping", train_data.class_indices) # Show happy=0, sad =1 vice versa

# 3. Build CNN model
model = Sequential([
    # conv block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D((2, 2)),
    # conv block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # conv block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # dense layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    # Output layer: sigmoid since only 2 classes 
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train  model

history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data
)

# Accuracy scores
print("final training accuracy:", history.history['accuracy'][-1])
print("final validation accuracy:", history.history['val_accuracy'][-1])

#5. Save trained model

model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
