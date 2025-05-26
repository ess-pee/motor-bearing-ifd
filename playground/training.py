import json

import tensorflow as tf
import keras
from data_preprocessing import preprocess


# Check if GPU is available
print(tf.config.list_physical_devices('GPU'))

# You shouldn't be here, I've already done the dirty work for you, the pretrained model is already in the directory.
# alright you can have a look at the model architecture, but there's a better description of that in the README
# and yes most of these hyperparameters are from heuristic search and standard data science practice.
# maybe I could have used a random search or a grid search algorithm but the validation accuracy is already 99.8% so...

if __name__ == "__main__":

    # I'm not a patient man, if training saturated we can stop early
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # choose the dataset you want to use, cwru, mfd or tri
    DATASET = 'tri'

    print("Starting to pull data...")
    xtrain, ytrain, xtest, ytest, classes, window_size, enc_ord = preprocess(DATASET)
    print("Data preprocessing completed successfully!, defining model...")

    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(window_size,1)))

    model.add(keras.layers.Conv1D(filters=128, kernel_size=9, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Conv1D(filters=64, kernel_size=9, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Conv1D(filters=32, kernel_size=9, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(len(classes)))
    model.add(keras.layers.Softmax())

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("Model defined successfully!, starting training...")
    history = model.fit(xtrain, ytrain, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
    print("Training completed successfully!")

    model.save(f'models/{DATASET}_model.h5')
    print("Model saved successfully!")

    with open(f'models/{DATASET}_encoding.json', 'w', encoding='utf-8') as f:
        json.dump(enc_ord, f)