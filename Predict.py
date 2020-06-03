import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras

import GlobalVariables
import Dataset
import Display


test = Dataset.getValidationDatasetGenerator()
test_images, test_labels = test.next()

model = keras.models.load_model('model.h5')
model.summary()

probability_model = keras.Sequential([model])

predictions = probability_model.predict_proba(test_images)

Display.TestResults(predictions, test_labels, test_images)

