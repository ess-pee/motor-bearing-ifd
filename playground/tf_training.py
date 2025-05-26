import tensorflow as tf
import keras
import numpy as np

from data_preprocessing import preprocess


# Check if GPU is available and set memory growth
print(tf.config.list_physical_devices('GPU'))

# Okay let's do some transfer learning, the pretrained model is already in the directory.
# So essentially what's happening here is that I'm taking the pretrained model and removing the last 7 layers, which are the dense classiification layers.
# Then I freeze the middle 2 convolutional layers and add new dense layers to the end of the model.
# However because we're training on a different dataset, we unfreeze the first 2 convolutional layers to let them learn domain specific features low level features.
# The rest of the model is frozen to prevent overfitting and to speed up training time.
# I'm getting about a 5-10% increase in accuracy over just having a fresh model trained on the limited dataset.

if __name__ == "__main__":

    # hehe funny number
    np.random.seed(69)

    # I'm not a patient man, if training saturated we can stop early
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    DATASET = 'tri' # choose the dataset you want to use, cwru, mfd or tri
    MODEL = 'cwru' # choose the model you want to use, cwru, mfd or tri
    PERCENT_SAMPLE = 0.1 # choose the percentage of the dataset you want to use for training, 0.05 is 5%

    print("Starting to pull data...")
    xtrain, ytrain, xtest, ytest, classes, window_size = preprocess(DATASET)

    # it wouldn't be transfer learning if I didn't sample the data into a smaller dataset so my experiment has some validity
    indices = np.random.choice(xtrain.shape[0], size=int(len(xtrain)*PERCENT_SAMPLE), replace=False)
    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    print("Data preprocessing completed successfully!, defining model...")


    print(f"Loading pretrained model from models/{MODEL}_model.h5")
    pretrained_model = keras.models.load_model(f'models/{MODEL}_model.h5')
    print("Pretrained model loaded successfully!")

    print("Beginning to instantiate new model...")
    transfer_model = keras.models.Sequential()
    transfer_model.add(keras.layers.Input(shape=(window_size,1)))

    print("Beheading the pretrained model and adding new layers...")
    for layer in pretrained_model.layers[:-7]:
        layer.trainable = False
        transfer_model.add(layer)

    for layer in transfer_model.layers[0:2]:
        layer.trainable = True

    transfer_model.add(keras.layers.Flatten())

    transfer_model.add(keras.layers.Dense(32, activation='relu'))
    transfer_model.add(keras.layers.Dropout(0.4))

    transfer_model.add(keras.layers.Dense(16, activation='relu'))
    transfer_model.add(keras.layers.Dropout(0.2))

    transfer_model.add(keras.layers.Dense(len(classes)))
    transfer_model.add(keras.layers.Softmax())

    opt = keras.optimizers.Adam(learning_rate=0.001)
    transfer_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("Model defined successfully!, starting training...")
    history = transfer_model.fit(xtrain, ytrain, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
    print("Training completed successfully!")

    print("Saving model...")
    transfer_model.save('models/transfer_model.h5')
    print("Model saved successfully!")
