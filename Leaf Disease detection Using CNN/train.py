import os
import sys
import io

os.environ['PYTHONIOENCODING'] = 'UTF-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

# Basic CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=None,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')
labels = (training_set.class_indices)
print(labels)

val_set = val_datagen.flow_from_directory('dataset/val',
                                          target_size=(128, 128),
                                          batch_size=32,
                                          class_mode='categorical')

labels2 = (val_set.class_indices)
print(labels2)

model.fit(training_set,
          steps_per_epoch=375,
          epochs=250,
          validation_data=val_set,
          validation_steps=125)

# Part 3 - Making new predictions

model_json = model.to_json()
with open('model.json', 'w', encoding='utf-8') as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')
print("Saved Model to Disk")