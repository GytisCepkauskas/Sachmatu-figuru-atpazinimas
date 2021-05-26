'''
2.8.3 skyriaus „Pasukimo metodas“ implementacija

Paleidimas:
python RotationMethod.py --sum-rule
python RotationMethod.py --best-prediction-rule

Reikalingi GlobalVariables kintamieji:
MODEL_TO_EVALUATE_FILE  -  modelio, kuriam bus pritaikytas pasukimo metodas, failas
ROTATION_RULE_ALPHA  -  alfa kampo dydis (laipsniais)

Pastabos:
Implementacija yra parašyta ypač neefektyviai, tačiau efektyvumas nėra svarbus darbo kontekste
'''

import sys
import math
import numpy

import keras
import tensorflow_addons as tfa

import GlobalVariables
import Dataset


def SumRule(model, images, labels, rotation_jumps):
    if (360 % rotation_jumps != 0):
        raise Exception("Invalid GlobalVariables.ROTATION_RULE_ALPHA value")

    turns = int(360 / rotation_jumps)
    correct_count = 0

    for index in range(len(images)):
        predictions = [0] * 6
        image = images[index]

        for i in range(turns):
            modifiedImage = tfa.image.rotate(image, math.radians(rotation_jumps * i), interpolation='BILINEAR', fill_mode="reflect")
            modifiedImage = numpy.expand_dims(modifiedImage, axis=0)
            prediction = model.predict_proba(modifiedImage)

            for classIndex in range(6):
                predictions[classIndex] += prediction[0][classIndex]

        correct_count += labels[index][numpy.argmax(predictions)]

    return correct_count / len(images)




def BestPredictionRule(model, images, labels, rotation_jumps):
    if (360 % rotation_jumps != 0):
        raise Exception("Invalid GlobalVariables.ROTATION_RULE_ALPHA value")

    turns = int(360 / rotation_jumps)
    correct_count = 0

    for index in range(len(images)):
        image = images[index]

        best_label = 0
        best_prediction = 0

        for i in range(turns):
            modifiedImage = tfa.image.rotate(image, math.radians(rotation_jumps * i), interpolation='BILINEAR', fill_mode="reflect")
            modifiedImage = numpy.expand_dims(modifiedImage, axis=0)
            prediction = model.predict_proba(modifiedImage)

            if (prediction[0][numpy.argmax(prediction[0])] > best_prediction):
                best_label = numpy.argmax(prediction[0])
                best_prediction = prediction[0][numpy.argmax(prediction[0])]

        correct_count += labels[index][best_label]

    return correct_count / len(images)



def EvaluateUsingRotationMethod():
    import csv

    model = keras.models.load_model(GlobalVariables.MODEL_TO_EVALUATE_FILE)
    probability_model = keras.Sequential([model])
    _, input_height, imput_length, input_depth = probability_model.layers[0].input_shape
    probability_model.compile(optimizer='adam',
                          loss=keras.losses.mean_squared_error,
                          metrics=['accuracy'])

    with open('Training_histories/RotationMethodResults.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['turn', 'acc'])
        writer.writeheader()
        
        for i in range(181):
            test = Dataset.getTestingDatasetGenerator(input_height, imput_length, isGrayscale=input_depth == 1, fixedRotationAngle=i)
            testImages, testLabels = test.next()
            if (sys.argv[1] == '-s' or sys.argv[1] == '--sum-rule'):
                result =  SumRule(probability_model, testImages, testLabels, GlobalVariables.ROTATION_RULE_ALPHA)
            elif (sys.argv[1] == '-b' or sys.argv[1] == '--best-prediction-rule'):
                result =  BestPredictionRule(probability_model, testImages, testLabels, GlobalVariables.ROTATION_RULE_ALPHA)
            else:
                raise Exception("Incorrect parameters passed")

            print (str(i) + "°:", str(result))

            writer.writerow({'turn': i, 'acc': result})


EvaluateUsingRotationMethod()