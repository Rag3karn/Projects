from numpy import loadtxt
from keras.models import model_from_json # type: ignore

dataset = loadtxt('pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:,0:8]
Y = dataset[:,8]

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.weights.h5')
print('Loaded Model from disk')

predictions = model.predict_step(X)
for i in range(5,10):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))