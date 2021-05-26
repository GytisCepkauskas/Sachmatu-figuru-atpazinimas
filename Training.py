'''
Tinklų mokymai

Paleidimas:
python Training.py
python Training.py --transfer-learning
python Training.py --fine-tuning

Reikalingi GlobalVariables kintamieji:
MODEL_TO_USE  -  mokomo modelio pavadinimas
DROPOUT_CHANCE  -  Praretinimo tikimybė (0 - 1.0) (Veikia tik su ChessNet ir OverfittingModel)
ROTATION_RULE_ALPHA  -  alfa kampo dydis (laipsniais)
FULL_ROTATION_ON  -  ar mokymo ir testavimo vaizdai turėtų būti pasukami 360 laispnių kampais
MODEL_TO_FINE_TUNE_FILE  -  modelio, kuriams bus pritaikoma adaptacija, failas

Pastabos:
Dėl saugumo saugomi tik mokymo metu geriausi modeliai (darbo rašymo metu buvo saugomi visi)
'''

import os
import sys
import time
import numpy
import random

import keras
import tensorflow

import Dataset
import GlobalVariables
import Display
import Models


seed_value = 789456123

os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
numpy.random.seed(seed_value)
tensorflow.random.set_seed(seed_value)


def TraditionalTraining():
    start_time = time.time()

    model = Models.GetEmptyByName(GlobalVariables.MODEL_TO_USE)
    model.summary()

    model.compile(optimizer="adam",
                loss=keras.losses.mean_squared_error,
                metrics=['accuracy'])

    image_height, image_length, color_depth = Models.Shapes[GlobalVariables.MODEL_TO_USE]
    history = model.fit(
        Dataset.getTrainingDatasetGenerator(image_height, image_length, color_depth == 1),
        epochs=GlobalVariables.EPOCHS,
        validation_data=Dataset.getTestingDatasetGenerator(image_height, image_length, color_depth == 1),
        callbacks=[
                   keras.callbacks.ModelCheckpoint("D:/Trained_models/{epoch:02d}e-{accuracy:.4f}-{val_accuracy:.4f}.h5", monitor='val_accuracy', save_best_only=False, verbose=0),
                   keras.callbacks.CSVLogger('Training_histories/Training.csv', append=True, separator=';')
                  ]
    )

    print ("Training time in sec:", time.time() - start_time)
    Display.TrainingHistory(history)


def TransferedLearning():
    start_time = time.time()

    model = Models.GetTrainedWithImageNetByName(GlobalVariables.MODEL_TO_USE)
    model.summary()

    model.compile(optimizer='adam',
                loss=keras.losses.mean_squared_error,
                metrics=['accuracy'])

    image_height, image_length, color_depth = Models.Shapes[GlobalVariables.MODEL_TO_USE]
    history = model.fit(
        Dataset.getTrainingDatasetGenerator(image_height, image_length, color_depth == 1),
        epochs=GlobalVariables.EPOCHS,
        validation_data=Dataset.getTestingDatasetGenerator(image_height, image_length, color_depth == 1),
        callbacks=[
                   keras.callbacks.ModelCheckpoint("D:/Trained_models/{epoch:02d}e-{accuracy:.4f}-{val_accuracy:.4f}.h5", monitor='val_accuracy', save_best_only=True, verbose=0),
                   keras.callbacks.CSVLogger('Training_histories/Test.csv', append=True, separator=';')
                  ]
    )

    print ("Training time in sec:", time.time() - start_time)
    Display.TrainingHistory(history)


def FineTuning(model = None):
    start_time = time.time()

    if (model == None):
        model = keras.models.load_model(GlobalVariables.MODEL_TO_FINE_TUNE_FILE)

    layer_count = len(model.layers)
    for index in range(0, layer_count):
        model.layers[index].trainable = False if index < layer_count * 0.92 else True

    model.summary()

    image_height, image_length, color_depth = Models.Shapes[GlobalVariables.MODEL_TO_USE]
    isGrayscale = color_depth == 1

    model.compile(optimizer='adam',
                loss=keras.losses.mean_squared_error,
                metrics=['accuracy'])

    history = model.fit(
        Dataset.getTrainingDatasetGenerator(image_height, image_length, isGrayscale),
        epochs=GlobalVariables.EPOCHS,
        validation_data=Dataset.getTestingDatasetGenerator(image_height, image_length, isGrayscale),
        callbacks=[
                   keras.callbacks.ModelCheckpoint("D:/Trained_models/{epoch:02d}e-{accuracy:.4f}-{val_accuracy:.4f}.h5", monitor='val_accuracy', save_best_only=True, verbose=0),
                   keras.callbacks.CSVLogger('Training_histories/Test.csv', append=True, separator=';')
                  ]
    )

    print ("Training time in sec:", time.time() - start_time)
    Display.TrainingHistory(history)


if (len(sys.argv) < 2):
    TraditionalTraining()
elif (sys.argv[1] == '-t' or sys.argv[1] == '--transfer-learning'):
    TransferedLearning()
elif (sys.argv[1] == '-f' or sys.argv[1] == '--fine-tuning'):
    FineTuning()
else:
    raise Exception("Incorrect parameters passed")