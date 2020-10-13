from tensorflow.keras import datasets, layers, models


def train_model(x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # 10 possible Classes
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Epochs is how many times the model is going to see the same data
    # Training the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)

    model.save('image_classifier.model')


