import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import toml

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

def main():
    parser = argparse.ArgumentParser(
        description='Train network based on data file')
    parser.add_argument("--data_file_name", required=True, nargs="+",
        help="set where the training data comes from.")
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    parser.add_argument("--alpha", required=True, 
        help="set alpha value of the Tikhonov Matrix")
    args = parser.parse_args()
    conf = toml.load(args.config)
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])

    data_files = args.data_file_name
    train_examples = np.zeros((1, 1, x_bins))
    print(train_examples.shape)
    train_labels = np.zeros((1, l_e_bins*l_y_bins))
    test_examples = np.zeros((1, 1, x_bins))
    test_labels = np.zeros((1,l_e_bins*l_y_bins))
    name_string = ""
    for i in range(len(data_files)):
        data_file = data_files[i]
        name_string = name_string+data_file.replace("/", "")+args.alpha
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

    print("train_examples.shape, train_labels.shape", train_examples.shape, train_labels.shape)

    #normalization u
    train_shape = train_examples.shape
    test_shape = test_examples.shape

    ratio = 0.01 #1.5e7

    max_train = 1e-9#1.5e-9#np.max(train_examples.reshape(train_shape[0], train_shape[1]*train_shape[2]), axis = 1)
    max_test = 1e-9#1.5e-9#np.max(test_examples.reshape(test_shape[0], test_shape[1]*test_shape[2]), axis = 1)
    train_examples = train_examples/max_train#[:, np.newaxis, np.newaxis]    
    train_labels = train_labels/ratio#max_train#[:, np.newaxis]  #units??
    test_examples = test_examples/max_test#[:, np.newaxis, np.newaxis] 
    test_labels = test_labels/ratio#max_test#[:, np.newaxis]
    print("train_examples[1][0][1]", train_examples[1][0][1])
    print("train_labels[1][1]", train_labels[1][1])

    print(test_examples.shape, test_labels.shape)
    
    
    #inverse_machine = np.zeros((l_e_bins*l_y_bins, x_bins*y_bins))
    test_examples_plot = np.squeeze(test_examples)
    img = (test_examples_plot)[:x_bins*1].T #x_bins*y_bins by x_bins*y_bins 
    spec = (train_labels)[:x_bins*1].T
    print("spec", spec.shape)
    plt.plot(spec)
    plt.savefig("all-test-labels-matrix-"+name_string.replace("/", "")+".png")
    plt.clf()
    print("img", img.shape)
    plt.plot(img)
    plt.savefig("all-test-examples-matrix"+name_string.replace("/", "")+".png")
    plt.clf()
    # inverse_machine img = spec want to find inverse_machine
    # inverse_machine = spec img^-1 but img may not be invertible
    # img^t inverse_machine^t = spec^t
    # A x = b
    # x =(A^t A+\Gamma^t\Gamma)^{-1}A^t b
    gamma = np.identity(x_bins*1)*float(args.alpha)
    print("gamma", gamma.shape)
    intermediate = np.dot(img, img.T)+np.dot(gamma.T, gamma)
    inverse_machine = np.dot(np.dot(np.linalg.inv(intermediate), img), spec.T)


    #print(model.evaluate(test_dataset))
    #print("Trained model, accuracy: {:5.2f}%".format(100*acc))

    test_predictions = np.dot(inverse_machine,(np.squeeze(test_examples).T)).T
    print("test_predictions.shape", test_predictions.shape)
    print("test_labels.shape", test_labels.shape)
    labels_max = np.argmax(test_labels, axis=1)
    print("labels_max.shape", labels_max.shape)
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
    plt.savefig(name_string+"-matrix.png")


if __name__ == "__main__":
    main()
