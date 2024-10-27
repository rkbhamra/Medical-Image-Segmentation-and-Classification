# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import utils


# Set UTF-8 encoding for standard output
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_height = 512
img_width = 512
class_names = ['healthy lung', 'tuberculosis lung']
folder = 'train'

# Load images and labels
x_train, y_train = utils.get_images(folder, img_width, img_height)
x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = utils.get_images(folder, img_width, img_height)
x_test, y_test = np.array(x_test), np.array(y_test)

print(f'data size :: {len(x_train)}')

model = models.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save('models/tuberculosis_model.keras')

# display images from training dataset
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):  # only take first batch
#     for i in range(min(9, images.shape[0])):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#
# plt.show()


'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
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
# plt.figure(figsize=(10, 10))
# for images, labels in test_ds.take(1):  # only take first batch
#     for i in range(min(9, images.shape[0])):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(f'Predicted: {class_names[predicted_classes[i]]}, True: {class_names[true_labels[i]]}' if true_labels is not None else f'Predicted: {class_names[predicted_classes[i]]}')
#         plt.axis("off")
#
# plt.show()

# this is bad: Test accuracy: 51.04%

'''

