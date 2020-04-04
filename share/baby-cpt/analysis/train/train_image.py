import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
import toml

def build_model(in_x, in_y, out_x, out_y):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(in_x, in_y)),
        tf.keras.layers.Dense(int(in_x/2*in_y), activation='relu'),
        tf.keras.layers.Dense(int(out_x*out_y*4), activation='relu'),
        tf.keras.layers.Dense(out_x*out_y)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train network based on data file')
    parser.add_argument("--data_file_name", required=True, nargs="+",
        help="set where the training data comes from.")
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    args = parser.parse_args()
    conf = toml.load(args.config)
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])

    data_files = args.data_file_name
    train_examples = np.zeros((1, x_bins, y_bins))
    train_labels = np.zeros((1, l_e_bins*l_y_bins))
    test_examples = np.zeros((1, x_bins, y_bins))
    test_labels = np.zeros((1,l_e_bins*l_y_bins))
    name_string = ""
    for i in range(len(data_files)):
        data_file = data_files[i]
        name_string = name_string+data_file.replace("/", "")
        with np.load(data_file+'.npz') as data:
            print(data['train_data'].shape)
            print(data['train_labels'].shape)
            train_examples = np.append(train_examples, data['train_data'], axis = 0 )
            train_labels = np.append(train_labels, data['train_labels'], axis = 0)
            test_examples = np.append(test_examples, data['test_data'], axis = 0)
            test_labels = np.append(test_labels, data['test_labels'], axis = 0)
            print(data_file, train_examples.shape, train_labels.shape)

    train_examples = train_examples[1:].astype(float)
    train_labels = train_labels[1:].astype(float)
    test_examples = test_examples[1:].astype(float)
    test_labels = test_labels[1:].astype(float)

    print(train_examples.shape, train_labels.shape)

    #normalization 
    train_shape = train_examples.shape
    test_shape = test_examples.shape
    max_train = np.max(train_examples.reshape(train_shape[0], train_shape[1]*train_shape[2]), axis = 1)
    max_test = np.max(test_examples.reshape(test_shape[0], test_shape[1]*test_shape[2]), axis = 1)
    print("max train", max_train[1])  
    train_examples = train_examples/max_train[:, np.newaxis, np.newaxis]    
    train_labels = train_labels/max_train[:, np.newaxis]  #units??
    test_examples = test_examples/max_test[:, np.newaxis, np.newaxis] 
    test_labels = test_labels/max_test[:, np.newaxis]
    print(train_examples[1][0][1])
    print(train_labels[1][1])
    #return 
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 400000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model = build_model(x_bins, y_bins, l_y_bins, l_e_bins)
    print(model.summary())

    checkpoint_path = "models/"+name_string+".ckpt"
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
    model.fit(train_dataset, epochs=10000, validation_data=test_dataset, 
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
    plt.hist2d(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1), 
        range=[[0, l_e_bins*l_y_bins], [0, l_y_bins*l_e_bins]], bins=50)
    plt.xlabel('True Max Index')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,l_y_bins*l_e_bins])
    plt.ylim([0,l_y_bins*l_e_bins])
    for x in range(6):
        #print(-100+(x-3)*31*5)
        #plt.plot([-100+(x-3)*31, 31*50+(x-3)*31], [-100-(x-3)*31, 31*50-(x-3)*31], c='red')
        plt.plot([-100, l_y_bins*l_e_bins], [-100, l_y_bins*l_e_bins], c='red')
    plt.savefig(name_string+".png")


if __name__ == "__main__":
    main()
