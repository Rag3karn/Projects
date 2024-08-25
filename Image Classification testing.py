from keras.models import model_from_json # type: ignore
import numpy as np
from keras.preprocessing import image # type: ignore
import os
import sys

# Configure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load the model
with open('model.json', 'r', encoding='utf-8') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('model.weights.h5')
print('Loaded Model from Disk')

def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] == 1:
        prediction = 'Thanos'
    else:
        prediction = 'Joker'

    print(prediction, img_name)

path = 'Dataset\\test'
files = []
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith('.jpg'):
            files.append(os.path.join(r, file))

if __name__ == "__main__":
    for f in files:
        classify(f)
        print('\n')
