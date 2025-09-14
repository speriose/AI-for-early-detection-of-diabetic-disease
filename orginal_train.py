import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define your folder structure
data_dir = r'C:\Users\Mohamed\Desktop\dataset_3\Diabetic retinopathy'
classes = ['Moderate', 'No_DR']

# Load and preprocess image data
def load_and_preprocess_data(data_dir, classes, target_shape=(64, 64)):
    data = []
    labels = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class directory '{class_dir}' does not exist.")

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Handle multiple formats
                file_path = os.path.join(class_dir, filename)
                # Load and preprocess the image
                img = load_img(file_path, target_size=target_shape)
                img_array = img_to_array(img)
                data.append(img_array)
                labels.append(i)

    if not data or not labels:
        raise ValueError("Dataset is empty. Ensure the data directory has valid images.")

    data = np.array(data) / 255.0  # Normalize images to [0, 1]
    labels = np.array(labels)

    print(f"Loaded {len(data)} images and {len(labels)} labels.")
    return data, labels

# Load dataset
data, labels = load_and_preprocess_data(data_dir, classes)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=len(classes))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a neural network model
input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=20, validation_data=(X_test, y_test))

# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy[1]}')

# Save the model
model.save('D_R.h5')
