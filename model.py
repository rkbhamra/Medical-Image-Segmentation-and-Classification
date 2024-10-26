import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys
import io

# Set UTF-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

img_height = 180  # Resize image to 180x180
img_width = 180
batch_size = 32   # For 1 epoch

# convert data directory into Path objects
data_dir = pathlib.Path("./train")
test_dir = pathlib.Path("./test")

image_count = len(list(data_dir.glob('*/**/*.png')))  
print(f"Total images in training directory: {image_count}")

#load training and validation data (80% train, 20% validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#load test data without any splitting since it is for testing only
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
print("Class names:", class_names)

'''
# display images from training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  # only take first batch
    for i in range(min(9, images.shape[0])):  
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

#plt.show()  
'''

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()



epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.show()



# After training the model, add this section to test the model on your test dataset.

# 1. Preprocess the test dataset
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. Make predictions on the test dataset
predictions = model.predict(test_ds)

# 3. Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=-1)

# 4. Get true labels (if available)
true_labels = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

# 5. Calculate accuracy (if true labels are available)
test_accuracy = np.mean(predicted_classes == true_labels)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# 6. Display results
# Visualize some test images with predictions
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):  # only take first batch
    for i in range(min(9, images.shape[0])):  
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f'Predicted: {class_names[predicted_classes[i]]}, True: {class_names[true_labels[i]]}' if true_labels is not None else f'Predicted: {class_names[predicted_classes[i]]}')
        plt.axis("off")

plt.show()

# this is bad: Test accuracy: 51.04%