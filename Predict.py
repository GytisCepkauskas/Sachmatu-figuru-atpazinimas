'''
Vizualiai parodo modelio spėjimus

Paleidimas:
python Predict.py

Reikalingi GlobalVariables kintamieji:
MODEL_TO_EVALUATE_FILE  -  modelio, kurio spėjimai bus parodyti, failas
'''

import keras

import GlobalVariables
import Dataset
import Display

model = keras.models.load_model(GlobalVariables.MODEL_TO_EVALUATE_FILE)
_, input_height, input_length, input_depth = model.layers[0].input_shape
model.summary()

test = Dataset.getTestingDatasetGenerator(input_height, input_length, input_depth == 1)
test_images, test_labels = test.next()

probability_model = keras.Sequential([model])

predictions = probability_model.predict_proba(test_images)

Display.TestResults(predictions, test_labels, test_images)

