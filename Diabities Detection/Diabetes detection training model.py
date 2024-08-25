#1.Number of times Pregnant
#2.Plasma Glucose Concentration a 2 hours in an oral glucose
#3.Diastolic blood Pressure (mm Hg)
#4.Triceps Skin fold Thickness (mm)
#5.2-Hour serum Insulin (mu m/ml)
#6.Body mass Index (weight in kg/(height in metre)^2)
#7.Diabetes Pedigree Function
#8.Age (in years)
#9.Class Function (0 or 1)

from numpy import loadtxt
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from keras.models import model_from_json # type: ignore
import sys

# Ensure the default encoding is set to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(X, Y, epochs=175, batch_size=10)

# Evaluate the Keras model
_, accuracy = model.evaluate(X, Y)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model to disk
model_json = model.to_json()
with open('model.json', "w", encoding='utf-8') as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')
print("Saved model to disk")

