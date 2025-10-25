import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load training and test CSVs
train_csv_path = 'C:/Users/abhin/Downloads/train.csv'
test_csv_path = 'C:/Users/abhin/Downloads/test.csv'

# Set the image size for MNIST-like data
image_size = 28

# Load CSVs
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Prepare X and y
X_train = train_df.drop('label', axis=1).values / 255.0
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values / 255.0
y_test = test_df['label'].values

# Reshape data to (samples, 28, 28, 1)
X_train = X_train.reshape(-1, image_size, image_size, 1)
X_test = X_test.reshape(-1, image_size, image_size, 1)

# One-hot encode labels
num_classes = max(y_train.max(), y_test.max()) + 1

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    validation_data=(X_test, y_test_cat),
    batch_size=32
)

# Save model
model.save('cnn_trained_csv_28x28.h5')
print("Training complete and model saved as cnn_trained_csv_28x28.h5")

