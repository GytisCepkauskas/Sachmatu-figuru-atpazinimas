import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.preprocessing.image import ImageDataGenerator

import GlobalVariables

def getDatasetGenerators():


    training_dir = os.path.join(".\\Dataset\\", 'Train')
    image_generator = ImageDataGenerator(rescale=1. / 255,
                                         horizontal_flip=True,
                                         rotation_range=15,
                                         zoom_range=0.15,
                                         width_shift_range=.2,
                                         height_shift_range=.2)

    train_data_gen = image_generator.flow_from_directory(batch_size=GlobalVariables.BATCH_SIZE,
                                                         directory=training_dir,
                                                         color_mode="grayscale",
                                                         shuffle=True,
                                                         target_size=(GlobalVariables.IMG_HEIGHT, GlobalVariables.IMG_WIDTH),
                                                         class_mode='categorical')
    return train_data_gen



def getValidationDatasetGenerator():

    validation_dir = os.path.join(".\\Dataset\\", "Validate")
    image_generator = ImageDataGenerator(rescale=1. / 255,
                                         horizontal_flip=True)

    validate_data_gen = image_generator.flow_from_directory(batch_size=GlobalVariables.BATCH_SIZE,
                                                            directory=validation_dir,
                                                            shuffle=True,
                                                            color_mode="grayscale",
                                                            target_size=(GlobalVariables.IMG_HEIGHT, GlobalVariables.IMG_WIDTH),
                                                            class_mode='categorical')
    return validate_data_gen