import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Conv2D, SpatialDropout2D, MaxPooling2D, Flatten, Dense, Dropout

import Dataset
import GlobalVariables
import Display


model = Sequential([
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
        Dense(6)
    ])

model.summary()


model.compile(optimizer='adam',
                #loss=keras.losses.categorical_crossentropy,
                #loss=keras.losses.sparse_categorical_crossentropy,
                loss=keras.losses.mean_squared_error,
                #loss=keras.losses.mean_squared_logarithmic_error,
                metrics=['accuracy'])


history = model.fit_generator(
    Dataset.getDatasetGenerators(),
    steps_per_epoch=GlobalVariables.TRAINING_SET_SIZE,
    epochs=GlobalVariables.EPOCHS,
    validation_data=Dataset.getValidationDatasetGenerator(),
    validation_steps=GlobalVariables.VALIDATION_SET_SIZE,
    callbacks=[keras.callbacks.ModelCheckpoint("Trained_models/{epoch:02d}e-{acc:.4f}-{val_acc:.4f}.h5", monitor='val_acc', verbose=1),
               keras.callbacks.CSVLogger('Final_3.csv', append=True, separator=',')]
)

model.save("model.h5")

Display.TrainingHistory(history)