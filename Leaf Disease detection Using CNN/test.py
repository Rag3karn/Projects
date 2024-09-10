# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing import image # type: ignore
from keras.models import model_from_json # type: ignore
import sys

# Ensure the console supports UTF-8
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# Load the model
json_file = open('model.json', 'r', encoding='utf-8')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")
print("Loaded model from disk")

# Define labels
labels = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
          "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
          "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
          "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
          "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
          "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
          "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
          "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]

# Define the actual label for the test image
actual_label = "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"  # Replace with the actual label of your test image

# Load an image for prediction
test_image_path = 'dataset\\test\\g.leafblight.JPG'  # Replace with your actual image file path
test_image = image.load_img(test_image_path, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = loaded_model.predict(test_image)
print(result)

# Find the label
label_index = result.argmax()
predicted_label = labels[label_index]
confidence = result[0][label_index]
print(f"Prediction: {predicted_label} with confidence {confidence}")

# Compare the predicted label with the actual label
if predicted_label == actual_label:
    print("The prediction is correct!")
else:
    print(f"The prediction is incorrect. The actual label is {actual_label}.")