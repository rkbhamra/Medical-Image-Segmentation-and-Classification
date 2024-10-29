import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import utils


def draw_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


def train_model(model_dir, x_data, y_data):
    train_ratio = 0.8
    x_train = x_data[:int(len(x_data) * train_ratio)]
    y_train = y_data[:int(len(y_data) * train_ratio)]
    x_validation = x_data[int(len(x_data) * train_ratio):]
    y_validation = y_data[int(len(y_data) * train_ratio):]

    print(f'data size :: {len(x_data)}')

    model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation))

    model.save(model_dir)
    with open(f'{model_dir}_history.json', 'w') as f:
        json.dump(history.history, f)


def load_model_history(model_dir):
    with open(f'{model_dir}.keras_history.json', 'r') as f:
        history = json.load(f)

        plt.plot(history['accuracy'], label='accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()
    return history


def test_model(model_dir, x_test, y_test):
    model = tf.keras.models.load_model(model_dir)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'test accuracy: {test_acc}')
    print(f'test loss: {test_loss}')


def use_model(model_dir, img_dir):
    model = tf.keras.models.load_model(model_dir)
    img = np.array([utils.get_image(img_dir, img_width, img_height)])
    prediction = model.predict(img)
    index = np.argmax(prediction)
    print('predictions :: ', prediction)
    print(f'accuracy :: {prediction[0][index] * 100:.2f}%')
    print(class_names[index])
    return class_names[index]


'''
******************************************************************************************************************
    will need the following to run:
    - test folder
    - train folder
    - models folder (empty if you want to train a new model, for testing should contain .keras model) 
    - pip install -r requirements.txt
    
    current model acc: 95.8% (92/96 images classified correctly)
******************************************************************************************************************
'''

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
img_height = 512
img_width = 512
class_names = ['healthy lung', 'tuberculosis lung']


# Load the data for training
# x_data, y_data = utils.get_images('train', img_width, img_height)

# Training
# train_model('models/tuberculosis_model.keras', x_data, y_data)
# history = load_model_history('models/tuberculosis_model')


# Load the data for testing
# x_test, y_test = utils.get_images('./test', img_width, img_height)

# Testing
# test_model('models/tuberculosis_model.keras', x_test, y_test)

# Use model
use_model('models/tuberculosis_model.keras', 'test/img/CHNCXR_0025_0.png')