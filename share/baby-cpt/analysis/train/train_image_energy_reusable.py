import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
import toml
import log_callback
from pbpl import common
import data_normalization

def build_model(conf): #in_x, in_y, out_x, out_y
    out_x = int(conf['PrimaryGenerator']['YBins'])
    out_y = int(conf['PrimaryGenerator']['EBins'])
    in_x = 1 #int(conf['Simulation']['YBins'])
    in_y = int(conf['Simulation']['XBins'])

    #Layers = [2048, 1024, 512, 512]
    #Biases = ["False", "False", "False", "False", "False"]
    #Activations = ["relu", "relu", "relu", "relu", "linear"]
    #LearningRate = 0.001

    layers = conf['NeuralNetwork']["Layers"]
    activations = conf["NeuralNetwork"]["Activations"]
    biases = conf["NeuralNetwork"]["Biases"]
    learning_rate = conf["NeuralNetwork"]["LearningRate"]
    sequence = [tf.keras.layers.Flatten(input_shape=(in_x, in_y))]
    i=0
    for i in range(len(layers)):
        use_bias = True
        if biases[i] == "False":
            use_bias = False 
        sequence.append(
            tf.keras.layers.Dense(int(layers[i]), use_bias=use_bias, activation=activations[i])
        )
    use_bias = True
    if biases[i] == "False":
        use_bias = False 
    sequence.append(
        tf.keras.layers.Dense(int(out_x*out_y), use_bias=use_bias, 
        activation=activations[i]))
    model = tf.keras.Sequential(sequence)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def plot_preview(edep, truth, desc):
    #edep_resized = edep.T
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(2, 2)

    if truth is not None:
        f3_ax1 = fig3.add_subplot(gs[0, 1])
        truth_im = f3_ax1.imshow(truth)
        f3_ax1.set_title("truth")
        f3_ax1.set_xlabel('E')
        f3_ax1.set_ylabel('Y')
        #fig3.colorbar(truth_im, cax=f3_ax1)
        f3_ax1 = fig3.add_subplot(gs[1, 1])
        truth_summed = np.sum(truth, axis = 0)
        f3_ax1.plot(truth_summed)
        f3_ax1.set_title("truth")
        f3_ax1.set_xlabel('E')

    f3_ax3 = fig3.add_subplot(gs[:, 0])
    f3_ax3.set_title('gs[:, 0]')
    #im = f3_ax3.imshow(edep_resized, origin='lower')
    #ax.clabel(CS, inline=1, fontsize=10)
    plot_e = edep#_resized.T
    f3_ax3.plot(plot_e[int(len(plot_e)/2)])
    f3_ax3.set_title("e dep")
    f3_ax3.set_xlabel('Y')
    f3_ax3.set_ylabel('Z')
    #fig3.colorbar(im, ax=f3_ax3)

    plt.savefig(desc)
    plt.clf()

def load_data(conf, data_files):
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])

    train_examples = np.zeros((1, 1, x_bins))
    print(train_examples.shape)
    train_labels = np.zeros((1, l_e_bins*l_y_bins))
    test_examples = np.zeros((1, 1, x_bins))
    test_labels = np.zeros((1,l_e_bins*l_y_bins))
    name_string = ""
    for i in range(len(data_files)):
        data_file = data_files[i]
        name_string = name_string+data_file.replace("/", "")
        with np.load(data_file+'.npz') as data:
            print(data['train_data'][:].shape)
            print(data['train_labels'].shape)
            train_examples = np.append(train_examples, data['train_data'][:, int(y_bins/2), np.newaxis], axis = 0 )#
            train_labels = np.append(train_labels, data['train_labels'], axis = 0)
            test_examples = np.append(test_examples, data['test_data'][:, int(y_bins/2), np.newaxis], axis = 0)#
            test_labels = np.append(test_labels, data['test_labels'], axis = 0)
            print(data_file, train_examples.shape, train_labels.shape)
            plot_preview(train_examples[-1], train_labels[-1, np.newaxis], "last-train-"+data_file.replace("/", "")+".png")
            plot_preview(test_examples[-1], test_labels[-1, np.newaxis], "last-test-"+data_file.replace("/", "")+".png")


    train_examples = train_examples[1:].astype(float)
    train_labels = train_labels[1:].astype(float)
    test_examples = test_examples[1:].astype(float)
    test_labels = test_labels[1:].astype(float)

    return (train_examples, train_labels, test_examples, test_labels, name_string)

def main(config): #config

    common.setup_plot()

    conf = toml.load(config)
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])

    data_files = conf['NeuralNetwork']["DataFileNames"]
    args_out = conf['NeuralNetwork']["ModelName"]
    train_examples, train_labels, test_examples, test_labels, name_string = load_data(conf, data_files)

    print("train_examples.shape, train_labels.shape", train_examples.shape, train_labels.shape)

    #normalization u
    train_shape = train_examples.shape
    test_shape = test_examples.shape

    train_examples, _ = data_normalization.normalize_examples(conf, train_examples)#/max_train/z_weights#[:, np.newaxis, np.newaxis]    
    train_labels = data_normalization.normalize_labels(conf, train_labels)#/ratio#max_train#[:, np.newaxis]  #units??
    test_examples, _ = data_normalization.normalize_examples(conf, test_examples)#/max_test/z_weights#[:, np.newaxis, np.newaxis] 
    test_labels = data_normalization.normalize_labels(conf, test_labels)#/ratio#max_test#[:, np.newaxis]
    print("train_examples[1][0][1]", train_examples[1][0][1])
    print("train_labels[1][1]", train_labels[1][1])

    model_folder = "models-grid"+"-"+args_out+"/"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    print(test_examples.shape, test_labels.shape)
    fig = plt.figure(figsize=(3+3/8, 2+2/8), dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(test_labels.T,alpha=0.1)
    ax.set_xlabel("index")
    ax.set_ylabel("energy density (arb)")
    plt.savefig(model_folder+"all-test-labels-"+name_string.replace("/", "")+"-"+args_out+".png")
    plt.clf()
    fig = plt.figure(figsize=(3+3/8, 2+2/8), dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    test_examples_plot = np.squeeze(test_examples)
    ax.plot(test_examples_plot.T, alpha=0.1)
    ax.set_xlabel("index")
    ax.set_ylabel("energy deposition (arb)")
    plt.savefig(model_folder+"all-test-examples-"+name_string.replace("/", "")+"-"+args_out+".png")
    plt.clf()

    labels_max = np.max(test_labels, axis=1)
    average_max = np.average(labels_max)
    print("averaged_label_max", average_max)
    print("max_label_max", np.max(labels_max))
    print("std_label_max", np.std(labels_max))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 400000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model = build_model(conf)
    print(model.summary())

    checkpoint_path = model_folder+name_string+".ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Loads the weights
    #model.load_weights(checkpoint_path)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    log_cb = log_callback.LogCallback(model_folder+"log.txt")

    # Train the model with the new callback
    model.fit(train_dataset, epochs=1000, validation_data=test_dataset, 
            callbacks=[cp_callback, early_stop, log_cb])#, prediction_cb

    #print(model.evaluate(test_dataset))
    #print("Trained model, accuracy: {:5.2f}%".format(100*acc))

    test_predictions = model.predict(test_examples)
    print(test_labels.shape)
    labels_max = np.argmax(test_labels, axis=1)
    print(labels_max.shape)
    print(labels_max[np.argmax(labels_max)])
    print(labels_max[np.argmin(labels_max)])
    plt.scatter(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1), s=4, alpha=0.3)#, range=[[0, 50*50], [0, 50*50]], bins=50*50)
    #plt.hist2d(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1), 
        #range=[[0, l_e_bins*l_y_bins], [0, l_y_bins*l_e_bins]], bins=50, cmap="gist_gray_r")
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
    plt.savefig(model_folder+name_string+"-"+args_out+".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train network based on data file')
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    args = parser.parse_args()
    main(args.config)
