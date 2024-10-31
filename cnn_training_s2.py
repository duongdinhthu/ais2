import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageFile
import datetime
import csv
import numpy as np

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to validate and remove bad images
def clean_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Removing bad file: {file_path}")
                os.remove(file_path)

# Clean up bad images in train and validation directories
clean_directory('data/train')
clean_directory('data/validation')

# Data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,  # Giảm độ xoay
    width_shift_range=0.05,  # Giảm chuyển động chiều rộng
    height_shift_range=0.05,  # Giảm chuyển động chiều cao
    shear_range=0.05,  # Giảm shear
    zoom_range=0.05,  # Giảm độ phóng đại
    horizontal_flip=True,  # Vẫn giữ lật ngang
    fill_mode='nearest'  # Cách lấp đầy pixel
)


validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=128,  # Consider reducing if needed
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=128,  
    class_mode='categorical'
)

# Define the EfficientNetB0 model with fine-tuning
checkpoint_path = 'best_model_optimized.keras'

if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)
else:
    print("No checkpoint found, training from scratch.")
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),  # Adjusted neurons
        BatchNormalization(),
        Dropout(0.3),  # Adjusted dropout
        Dense(13, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),  # Adjusted learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Custom callback to log metrics and save the best validation accuracy
class MetricsCallback(Callback):
    def __init__(self, filename='metrics.csv', best_acc_file='best_val_accuracy.txt'):
        super(MetricsCallback, self).__init__()
        self.filename = filename
        self.best_acc_file = best_acc_file
        self.best_val_accuracy = 0.0

        if os.path.exists(self.best_acc_file):
            with open(self.best_acc_file, 'r') as f:
                line = f.readline()
                self.best_val_accuracy = float(line.split()[-1])
                print(f"Loaded best validation accuracy: {self.best_val_accuracy}")

        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        train_loss = logs.get('loss')
        train_accuracy = logs.get('accuracy')
        val_loss = logs.get('val_loss')

        if val_accuracy and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            with open(self.best_acc_file, 'w') as f:
                f.write(f'Best validation accuracy: {self.best_val_accuracy:.4f}\n')
            print(f"Updated best validation accuracy: {self.best_val_accuracy}")

        self.writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy])

    def on_train_end(self, logs=None):
        self.file.close()

# Callbacks: Early Stopping, Model Checkpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

metrics_callback = MetricsCallback()

# Training the model with the new configuration
model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr, metrics_callback]
)

# Save the best model after training in SavedModel format
model.save(checkpoint_path)  # Save the model with .keras extension
