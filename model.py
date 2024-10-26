from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_classification_model():
    model = Sequential([
        # first convolutional layer for leo's mom
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        # secons convolutional layer for leo's mom
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # third convolutional layer for leo's mom
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # flatten the 3D array to 1D array
        Flatten(),
        # connect the 1D array to the fully connected layer
        Dense(128, activation='relu'),
        # drop out 50% of the neurons (like leo did)...to prevent overfitting
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # binary classification
    ])
    return model

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
