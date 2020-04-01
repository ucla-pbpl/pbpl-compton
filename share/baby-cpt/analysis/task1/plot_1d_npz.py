import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Plot 1d Data')
    parser.add_argument("--data_files", required=True, nargs="+",
        help="set where the data comes from.")
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    args = parser.parse_args()

    name_string = ""
    for i in range(len(args.data_files)):
        data_file = args.data_files[i]
        name_string = name_string+data_file.replace("/", "")
        with np.load(data_file) as data:
            prediction = np.array(data["prediction"])
            prediction_1d = np.sum(prediction, axis=0)
            plt.plot(prediction_1d, label = data_file)
    plt.legend()
    #plt.show()
    plt.savefig(name_string+".png")

if __name__ == "__main__":
    main()

