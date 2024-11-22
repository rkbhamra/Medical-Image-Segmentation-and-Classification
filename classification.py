import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

    # Define the data augmentation for the training set
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest'
    )

    # Validation data does not need augmentation but should be normalized
    val_datagen = ImageDataGenerator()

    for fold, (xi, yi) in enumerate(skf.split(x_data, y_data)):
        print(f"Training fold {fold + 1}/{k_folds}...")
        x_train, x_validation = x_data[xi], x_data[yi]
        y_train, y_validation = y_data[xi], y_data[yi]
        print(f"Train size: {len(x_train)} Validation size: {len(x_validation)}")

        # Create the image data generators for this fold
        train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
        validation_generator = val_datagen.flow(x_validation, y_validation, batch_size=32)

        model = models.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        model.summary()
        history = model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)

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

        # Plot training & validation loss values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Loss')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

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
    index = round(prediction[0][0])
    acc = prediction[0][0] if index == 1 else 1 - prediction[0][0]

    print('predictions :: ', prediction)
    print(f'accuracy :: {acc * 100:.2f}%')
    print(class_names[index])

    ui_img = utils.get_ui_output(img_dir, img_width, img_height, index)
    # draw_images(np.array([ui_img]), [index])

    return [class_names[index], acc, ui_img]


def init_training():
    # Load the data for training (https://datasetninja.com/chest-xray)
    x_data, y_data = utils.get_images('res/train/img', img_width, img_height)

    # Load the data for training (https://data.mendeley.com/datasets/8j2g3csprk/2)
    x_data2, y_data2 = utils.get_images('res/mendeley/healthy', img_width, img_height, True, 0, 500)
    x_data3, y_data3 = utils.get_images('res/mendeley/TB', img_width, img_height, True, 1, 500)

    # concatenate the data
    x_data = np.concatenate((x_data, x_data2, x_data3))
    y_data = np.concatenate((y_data, y_data2, y_data3))
    # x_data = np.concatenate((x_data2, x_data3))
    # y_data = np.concatenate((y_data2, y_data3))

    # Shuffle the data
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    print("y_data (100) :: ", y_data[:100])

    # Training
    train_model('models/tuberculosis_model.keras', x_data, y_data)


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

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
print("tensorflow version :: ", tf.__version__)
img_height = 128
img_width = 128
class_names = ['healthy lung', 'tuberculosis lung']

# TRAINING
init_training()

# Load the data for testing
# x_test, y_test = utils.get_images('res/example_data/img', img_width, img_height)
x_test, y_test = utils.get_images('res/test/img', img_width, img_height)

# Testing
test_model('models/tuberculosis_model.keras', x_test, y_test)

# Use model
use_model('models/tuberculosis_model.keras', 'res/example_data/img/CHNCXR_0336_1.png')
use_model('models/tuberculosis_model.keras', 'res/example_data/img/CHNCXR_0025_0.png')

load_model_history('models/tuberculosis_model')
