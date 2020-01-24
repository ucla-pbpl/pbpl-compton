import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from Geant4.hepunit import *

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(122, 128)),
        tf.keras.layers.Dense(64*128, activation='relu'),
        tf.keras.layers.Dense(64*64, activation='relu'),
        tf.keras.layers.Dense(50*50)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def norm_energy(X):
    return (X/1000)# / (2e6/8000*MeV)

def norm_photons(X):
    return (X) / (100)

def main():
    data_file = 'set0-right-y-mixed'
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []
    with np.load(data_file+'.npz') as data:
        train_examples = norm_energy(data['train_data'])
        train_labels = norm_photons(data['train_labels'])
        test_examples = norm_energy(data['test_data'])
        test_labels = norm_photons(data['test_labels'])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 400000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model = build_model()
    print(model.summary())

    checkpoint_path = "models/col-norm-cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Loads the weights
    #model.load_weights(checkpoint_path)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Train the model with the new callback
    model.fit(train_dataset, epochs=10, validation_data=test_dataset, 
            callbacks=[cp_callback, early_stop])

    #print(model.evaluate(test_dataset))
    #print("Trained model, accuracy: {:5.2f}%".format(100*acc))

    test_predictions = model.predict(test_examples)
    print(test_labels.shape)
    labels_max = np.argmax(test_labels, axis=1)
    print(labels_max.shape)
    print(labels_max[np.argmax(labels_max)])
    print(labels_max[np.argmin(labels_max)])
    #plt.scatter(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1))#, range=[[0, 50*50], [0, 50*50]], bins=50*50)
    plt.hist2d(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1), range=[[0, 50*50], [0, 50*50]], bins=50)
    plt.xlabel('True Max Index')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,50*50])
    plt.ylim([0,50*50])
    for x in range(6):
        #print(-100+(x-3)*31*5)
        #plt.plot([-100+(x-3)*31, 31*50+(x-3)*31], [-100-(x-3)*31, 31*50-(x-3)*31], c='red')
        plt.plot([-100, 50*50], [-100, 50*50], c='red')
    plt.savefig(data_file+"-normed.png")


if __name__ == "__main__":
    main()
