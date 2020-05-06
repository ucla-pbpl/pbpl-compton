import numpy as np

z_weights = (0.0006*np.linspace(0, 127, 128)**2+0.2)/5

def normalize_examples(conf, examples):
     #0-1-2
    #max_ex = 0.1e-9
    print(examples.shape)
    default_max_ex = conf['NeuralNetwork']["DefaultMaxEx"]
    print("data_normalization normalize: max_ex", default_max_ex)
    normalized = examples/default_max_ex/z_weights*80
    print("max normalized: ", np.max(normalized))
    center = 0#np.max(normalized)/2
    print(normalized.shape)
    return (normalized-center, default_max_ex)

def normalize_examples_indi(conf, examples): #n, 64, 128
    count = examples.shape[0]
    print(count)
    examples_flattened = examples.reshape(count, -1)#N, 128*64
    max_ex = np.max(examples_flattened.T, axis = 0) # N
    print(max_ex)
    examples_normalized = (examples_flattened.T)/max_ex #128*64, N
    weighted = examples_normalized.T.reshape(count, -1, 128)/z_weights*80
    print("max normalized: ", np.max(weighted))
    print(max_ex)
    return (weighted, max_ex)
    
def normalize_labels(conf, labels):
     #1.5e7
    default_ratio = conf['NeuralNetwork']["DefaultRatio"]
    return labels/default_ratio

def recover_labels(conf, predicted):
    print("data_normalization recover: max_ex", max_ex)
    default_ratio = conf['NeuralNetwork']["DefaultRatio"]
    labels=predicted*default_ratio
    print("max labels: ", np.max(labels))
    return labels

def recover_labels_indi(conf, predicted, max_ex): #N, 128 || N
    default_max_ex = conf['NeuralNetwork']["DefaultMaxEx"]
    default_ratio = conf['NeuralNetwork']["DefaultRatio"]
    print(predicted.shape)
    print("data_normalization recover: max_ex", max_ex)
    print("ratio ", max_ex/default_max_ex)
    labels=((predicted.T)*default_ratio*max_ex/default_max_ex).T
    print("max labels: ", np.max(labels))
    return labels
    
    