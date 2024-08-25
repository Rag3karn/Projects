from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import sys
import io

# Set default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define the model
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation and preparation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Dataset\\train', target_size=(64, 64), batch_size=8, class_mode='binary')
val_set = val_datagen.flow_from_directory('Dataset\\val', target_size=(64, 64), batch_size=8, class_mode='binary')

# Train the model
model.fit(training_set, steps_per_epoch=10, epochs=50, validation_data=val_set, validation_steps=2)

# Save the model
model_json = model.to_json()
with open('model.json', 'w', encoding='utf-8') as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')
print("Saved Model to Disk")
