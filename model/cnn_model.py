import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def create_CNN_model(X_train, y_train, X_test, y_test, X_val, y_val):
    # Reshape data
    X_train_flattened = X_train.reshape(len(X_train), 28*28)
    X_test_flattened  = X_test.reshape(len(X_test), 28*28)
    X_val_flattened  = X_val.reshape(len(X_val), 28*28)

    cnn = Sequential()
    cnn.add(Flatten())
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=10, activation='softmax'))

    callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    cnn.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    history = cnn.fit(x=X_train_flattened, y=y_train, validation_data=(X_val_flattened, y_val), epochs=600, batch_size=32, verbose=False, callbacks=[callback])

    test_loss, test_acc = cnn.evaluate(x=X_test_flattened, y=y_test)

    print(f"Test accuracy: {test_acc}")

    predictions = cnn.predict(X_test_flattened)

    return predictions