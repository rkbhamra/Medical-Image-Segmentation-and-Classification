import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
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


def train_model(model_dir, x_data, y_data, k_folds=5):
    print(f'data size :: {len(x_data)}')
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (xi, yi) in enumerate(skf.split(x_data, y_data)):
        print(f"Training fold {fold + 1}/{k_folds}...")
        x_train, x_validation = x_data[xi], x_data[yi]
        y_train, y_validation = y_data[xi], y_data[yi]

        model = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.summary()
        history = model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), verbose=1)

        model.save(model_dir)
        with open(f'{model_dir}_history.json', 'w') as f:
            json.dump(history.history, f)

        val_accuracy = history.history['val_accuracy'][-1]
        fold_accuracies.append(val_accuracy)
        print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average Validation Accuracy: {avg_accuracy:.4f}")


def load_model_history(model_dir):
    with open(f'{model_dir}.keras_history.json', 'r') as f:
        history = json.load(f)
        plt.plot(history['loss'], label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
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
    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    print('predictions :: ', prediction)
    print(f'accuracy :: {prediction[0][index] * 100:.2f}%')
    print(class_names[index])

    ui_img = utils.get_ui_output(img_dir, img_width, img_height, index)
    draw_images(np.array([ui_img]), [index])

    return [class_names[index], prediction[0][index], ui_img]


def init_training():
    # Load the data for training (https://datasetninja.com/chest-xray)
    # x_data, y_data = utils.get_images('res/train/img', img_width, img_height)

    # Load the data for training (https://data.mendeley.com/datasets/8j2g3csprk/2)
    x_data2, y_data2 = utils.get_images('res/mendeley/healthy', img_width, img_height, True, 0)
    x_data3, y_data3 = utils.get_images('res/mendeley/TB', img_width, img_height, True, 1)

    # Load the data but shifted in random directions and mirrored on the y axis
    # x_data4, y_data4 = np.concatenate((x_data, x_data2, x_data3)), np.concatenate((y_data, y_data2, y_data3))
    x_data4, y_data4 = np.concatenate((x_data2, x_data3)), np.concatenate((y_data2, y_data3))
    for i in range(len(x_data4)):
        xs, ys = np.random.choice([-30, 30], 2)
        x_data4[i] = np.flip(x_data4[i], 1)
        x_data4[i] = np.roll(x_data4[i], xs, axis=0)
        x_data4[i] = np.roll(x_data4[i], ys, axis=1)

    # concatenate the data
    # x_data = np.concatenate((x_data, x_data2, x_data3, x_data4))
    # y_data = np.concatenate((y_data, y_data2, y_data3, y_data4))
    x_data = np.concatenate((x_data2, x_data3, x_data4))
    y_data = np.concatenate((y_data2, y_data3, y_data4))

    # Shuffle the data
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]

    # Training
    train_model('models/tuberculosis_model.keras', x_data, y_data)
    load_model_history('models/tuberculosis_model')


'''
******************************************************************************************************************
    - pip install -r requirements.txt
    
    will need the following to train new model:
    - res/mendeley
    - res/train
    
    will need the following to use model:
    - models folder (contains .keras file) 
******************************************************************************************************************
'''

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("tensorflow version :: ", tf.__version__)
img_height = 128
img_width = 128
class_names = ['healthy lung', 'tuberculosis lung']

# TRAINING
# init_training()

# Load the data for testing
# x_test, y_test = utils.get_images('res/example_data/img', img_width, img_height)

# Testing
# test_model('models/tuberculosis_model.keras', x_test, y_test)

# Use model
# use_model('models/tuberculosis_model3.keras', 'lung.jpg')
use_model('models/tuberculosis_model3.keras', 'res/example_data/img/CHNCXR_0025_0.png')

# load_model_history('models/tuberculosis_model')
