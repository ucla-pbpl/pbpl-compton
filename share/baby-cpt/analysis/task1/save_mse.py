import glob
import os
import toml

potential_files = glob.glob('models-grid-*-N*/mse.txt')
print(potential_files)
with open("grand_mse_selected.csv", "w") as of:
    for pf in potential_files:
        with open(pf, "r") as f:
            model_name = os.path.basename(os.path.dirname(pf))
            tml_name = model_name.replace("models-grid", "config/task1")+".toml"
            conf = toml.load(tml_name)
            layer_max_count=4
            
            layers = list(conf['NeuralNetwork']['Layers'])
            biases = list(conf['NeuralNetwork']['Biases'])
            activations = list(conf['NeuralNetwork']['Activations'])
            layer_count = len(layers)
            if (layer_count!=2):
                continue
            for i in range(layer_count, layer_max_count):
                layers.append("")
            for j in range(layer_count, layer_max_count+1):
                biases.append("")
                activations.append("")
            of.write(model_name+",")
            of.write(f.readline()+",")
            of.write(",".join(layers))
            of.write(",")
            of.write(",".join(biases))
            of.write(",")
            of.write(",".join(activations))
            of.write(",")
            of.write(str(conf['NeuralNetwork']['LearningRate']))
            of.write(",")
            of.write(str(conf['NeuralNetwork']['DefaultMaxEx']))
            of.write(",")
            of.write(str(conf['NeuralNetwork']['DefaultRatio']))
            of.write("\n")
