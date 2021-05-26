'''
„Chessman image dataset“ duomenų rinkinio užkrovimas, suskaidymas į mokymo ir testinius duomenis, vaizdų augmentavimas
'''

import os
import math

import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator

import GlobalVariables

def getTrainingDatasetGenerator(image_height = GlobalVariables.IMG_HEIGHT, image_length = GlobalVariables.IMG_WIDTH, isGrayscale = True):
    if GlobalVariables.ENABLE_IMAGE_AUGMENTATION:
        image_generator = ImageDataGenerator(rescale=1. / 255,
                                             horizontal_flip=True,
                                             rotation_range = 180 if GlobalVariables.FULL_ROTATION_ON else 20,
                                             zoom_range=0.15,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             )
    else:
        image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = image_generator.flow_from_directory(batch_size=GlobalVariables.BATCH_SIZE,
                                                         directory=os.path.join(".\\Dataset\\", 'Train'),
                                                         color_mode= "grayscale" if isGrayscale else "rgb",
                                                         shuffle=True,
                                                         target_size=(image_height, image_length),
                                                         class_mode='categorical')
    return train_data_gen



def getTestingDatasetGenerator(image_height = GlobalVariables.IMG_HEIGHT, image_length = GlobalVariables.IMG_WIDTH, isGrayscale = True, fixedRotationAngle = GlobalVariables.FIXED_ROTATION_DEGREE):
    image_generator = ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=180 if GlobalVariables.FULL_ROTATION_ON else 0,
                                         preprocessing_function=lambda image : tfa.image.rotate(image, math.radians(fixedRotationAngle), interpolation='BILINEAR', fill_mode="reflect"),
                                        )

    validate_data_gen = image_generator.flow_from_directory(batch_size=GlobalVariables.BATCH_SIZE if GlobalVariables.BATCH_SIZE < GlobalVariables.VALIDATION_SET_SIZE else GlobalVariables.VALIDATION_SET_SIZE,
                                                            directory=os.path.join(".\\Dataset\\", "Validate"),
                                                            color_mode= "grayscale" if isGrayscale else "rgb",
                                                            shuffle=False,
                                                            # save_to_dir="D:\TEMP",
                                                            # save_prefix="0_",
                                                            # save_format="jpg",
                                                            target_size=(image_height, image_length),
                                                            class_mode='categorical')
    return validate_data_gen