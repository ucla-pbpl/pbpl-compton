import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(122, 128)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(31)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.0001)

    model.compile(loss='kld',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def main():
    data_file = 'set02ag1_image.npz'
    train_examples = []
    train_labels = []
    test_examples = []
    test_labels = []
    with np.load(data_file) as data:
        train_examples = data['train_data']
        train_labels = data['train_labels']
        test_examples = data['test_data']
        test_labels = data['test_labels']

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 400000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model = build_model()

    checkpoint_path = "models/image-kld-cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Loads the weights
    #model.load_weights(checkpoint_path)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    # Train the model with the new callback
    model.fit(train_dataset, epochs=10000, validation_data=test_dataset, 
            callbacks=[cp_callback, early_stop])

    #print(model.evaluate(test_dataset))
    #print("Trained model, accuracy: {:5.2f}%".format(100*acc))

    test_predictions = model.predict(test_examples)
    #print(test_labels.shape)
    plt.hist2d(np.argmax(test_labels, axis=0), np.argmax(test_predictions, axis=0),range=[[0, 30], [0, 30]], bins=30)
    plt.xlabel('True Values [MeV]')
    plt.ylabel('Predictions [MeV]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.plot([-100, 100], [-100, 100], c='red')
    plt.show()


if __name__ == "__main__":
    main()
