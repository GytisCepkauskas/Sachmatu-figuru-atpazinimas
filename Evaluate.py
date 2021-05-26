'''
Įvertina modelio tikslumą su testiniais duomenimis

Paleidimas:
python Evaluate.py
python Evaluate.py --full-rotation

Reikalingi GlobalVariables kintamieji:
MODEL_TO_EVALUATE_FILE  -  modelio, kurio tikslumas bus įvertintas, failas
'''

import sys

import keras

import GlobalVariables
import Dataset


def EvaluateAccuracy():
    model = keras.models.load_model(GlobalVariables.MODEL_TO_EVALUATE_FILE)
    model.summary()
    probability_model = keras.Sequential([model])
    _, input_height, imput_length, input_depth = probability_model.layers[0].input_shape

    test = Dataset.getTestingDatasetGenerator(input_height, imput_length, isGrayscale=input_depth == 1)

    probability_model.compile(optimizer='adam',
                            loss=keras.losses.mean_squared_error,
                            metrics=['accuracy',
                                     keras.metrics.TopKCategoricalAccuracy(k=2, name="Top_2_acc"),
                                     keras.metrics.TopKCategoricalAccuracy(k=3, name="Top_3_acc")])

    probability_model.evaluate(test, verbose=1)



def EvaluateRotationAccuracy(verbose = True):
    import csv

    model = keras.models.load_model(GlobalVariables.MODEL_TO_EVALUATE_FILE)
    probability_model = keras.Sequential([model])
    _, input_height, imput_length, input_depth = probability_model.layers[0].input_shape
    probability_model.compile(optimizer='adam',
                          loss=keras.losses.mean_squared_error,
                          metrics=['accuracy'])

    with open('Training_histories/RotationAccEvaluation.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['turn', 'loss', 'acc'])
        writer.writeheader()


        for i in range(181):
            test = Dataset.getTestingDatasetGenerator(input_height, imput_length, isGrayscale=input_depth == 1, fixedRotationAngle=i)

            if verbose:
                print (str(i) + "°:")
            
            result = probability_model.evaluate(test, verbose=1 if verbose else 0)

            writer.writerow({'turn': i, 'loss': result[0], 'acc': result[1]})



if (len(sys.argv) < 2):
    EvaluateAccuracy()
elif (sys.argv[1] == '-f' or sys.argv[1] == '--full-rotation'):
    EvaluateRotationAccuracy(verbose = True)
else:
    raise Exception("Incorrect parameters passed")