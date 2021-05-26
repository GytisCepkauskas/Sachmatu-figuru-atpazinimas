'''
Grafikų ir spėjimų vizualicija
'''

import numpy

import matplotlib.pyplot as pyplot

import GlobalVariables

TextLabels = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

def TrainingHistory(history):
    pyplot.figure(figsize=(8, 8))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(range(GlobalVariables.EPOCHS), history.history['accuracy'], label='Apmokymo tikslumas')
    pyplot.plot(range(GlobalVariables.EPOCHS), history.history['val_accuracy'], label='Validavimo tikslumas')
    pyplot.legend(loc='lower right')
    pyplot.title('Apmokymo ir validavimo tikslumai')

    pyplot.subplot(1, 2, 2)
    pyplot.plot(range(GlobalVariables.EPOCHS), history.history['loss'], label='Apmokymo nuostoliai')
    pyplot.plot(range(GlobalVariables.EPOCHS), history.history['val_loss'], label='Validavimo nuostoliai')
    pyplot.legend(loc='upper right')
    pyplot.title('Apmokymo ir validavimo nuostoliai')
    pyplot.show()


def TestResults(predictions, test_labels, test_images):
    print(len(test_images))
    num_images = len(test_images)
    num_rows = num_images / 6
    num_cols = 6
    print (predictions)
    pyplot.figure(figsize=(6*num_cols, 2*num_rows))
    for i in range(num_images):
      pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
      TestResult(i, predictions[i], test_labels[i], test_images[i])
    pyplot.tight_layout()
    pyplot.show()


def TestResult(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label, img
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.imshow(img.squeeze(), cmap=pyplot.cm.binary)

    predicted_label = numpy.argmax(predictions_array)
    
    print (predictions_array, predicted_label)

    label = str(TextLabels[predicted_label]) + " (" + str(predictions_array[predicted_label]) + ")"

    if true_label[predicted_label] == 1:
      pyplot.xlabel(label, color = 'green')
    else:
      pyplot.xlabel(label, color = 'red')


def Images(images_arr):
    fig, axes = pyplot.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img.squeeze(), cmap=pyplot.cm.binary)
        ax.axis('off')
    pyplot.tight_layout()
    pyplot.show()