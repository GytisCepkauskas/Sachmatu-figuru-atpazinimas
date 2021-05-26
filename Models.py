'''
Modelių struktūros ir įvesties formos informacija
'''

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, SpatialDropout2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

import GlobalVariables

def ChessNet():
    return Sequential([
        Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu', input_shape=(GlobalVariables.IMG_HEIGHT, GlobalVariables.IMG_WIDTH, 1)),
        SpatialDropout2D(GlobalVariables.DROPOUT_CHANCE),
        MaxPooling2D(),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        SpatialDropout2D(GlobalVariables.DROPOUT_CHANCE),
        MaxPooling2D(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        SpatialDropout2D(GlobalVariables.DROPOUT_CHANCE),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(GlobalVariables.DROPOUT_CHANCE),
        Dense(6, activation="softmax")
    ])

def OverfittingModel():
    return Sequential([
        Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu', input_shape=(GlobalVariables.IMG_HEIGHT, GlobalVariables.IMG_WIDTH, 1)),
        SpatialDropout2D(GlobalVariables.DROPOUT_CHANCE),
        MaxPooling2D(),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        SpatialDropout2D(GlobalVariables.DROPOUT_CHANCE),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(GlobalVariables.DROPOUT_CHANCE),
        Dense(6, activation="softmax")
    ])

Shapes = {
    "ChessNet": (256, 256, 1),
    "OverfittingModel": (256, 256, 1),
    "VGG19": (224, 224, 3),
    "InceptionV3": (299, 299, 3),
    "Xception": (299, 299, 3),
    "MobileNetV2": (224, 224, 3),
    "ResNet50V2": (224, 224, 3),
    "ResNet101V2": (224, 224, 3),
}

def GetEmptyByName(name):
    if name == "ChessNet":
        return ChessNet()
    if name == "OverfittingModel":
        return OverfittingModel()
    if name == "VGG19":
        return keras.applications.VGG19(weights=None, classes=6)
    elif name == "InceptionV3":
        return keras.applications.InceptionV3(weights=None, classes=6)
    elif name == "Xception":
        return keras.applications.Xception(weights=None, classes=6)
    elif name == "MobileNetV2":
        return keras.applications.MobileNetV2(weights=None, classes=6, alpha=1)
    elif name == "ResNet50V2":
        return keras.applications.ResNet50V2(weights=None, classes=6)
    elif name == "ResNet101V2":
        return keras.applications.ResNet101V2(weights=None, classes=6)
    else:
        raise ValueError('Name for unsupported model passed to Models.GetEmptyByName')

def GetTrainedWithImageNetByName(name):
    if GlobalVariables.MODEL_TO_USE == "VGG19":
        image_height, image_length, isGrayscale = 224, 224, False
        pretrained_model = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(image_height, image_length, 3))
    elif GlobalVariables.MODEL_TO_USE == "InceptionV3":
        image_height, image_length, isGrayscale = 299, 299, False
        pretrained_model = keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(image_height, image_length, 3))
    elif GlobalVariables.MODEL_TO_USE == "Xception":
        image_height, image_length, isGrayscale = 299, 299, False
        pretrained_model = keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(image_height, image_length, 3))
    elif GlobalVariables.MODEL_TO_USE == "MobileNetV2":
        image_height, image_length, isGrayscale = 224, 224, False
        pretrained_model = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_height, image_length, 3))
    elif GlobalVariables.MODEL_TO_USE == "ResNet50V2":
        image_height, image_length, isGrayscale = 224, 224, False
        pretrained_model = keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(image_height, image_length, 3))
    elif GlobalVariables.MODEL_TO_USE == "ResNet101V2":
        image_height, image_length, isGrayscale = 224, 224, False
        pretrained_model = keras.applications.ResNet101V2(include_top=False, weights="imagenet", input_shape=(image_height, image_length, 3))
    else:
        raise ValueError('Name for unsupported model passed to Models.GetTrainedWithImageNetByName')

    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(6, activation="softmax")(x)

    model = Model(inputs=pretrained_model.input, outputs=x)

    pretrained_model.trainable = False

    return model