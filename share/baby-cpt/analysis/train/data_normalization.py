import numpy as np

z_weights = (0.0006*np.linspace(0, 127, 128)**2+0.2)/5
default_max_ex=0.1e-9*80


def normalize_examples(examples):
     #0-1-2
    #max_ex = 0.1e-9
    print(examples.shape)
    print("data_normalization normalize: max_ex", default_max_ex)
    normalized = examples/default_max_ex/z_weights*80
    print("max normalized: ", np.max(normalized))
    center = 0#np.max(normalized)/2
    print(normalized.shape)
    return (normalized-center, default_max_ex)

def normalize_examples_indi(examples): #n, 64, 128
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
    
def normalize_labels(labels):
    ratio = 0.01 #1.5e7
    return labels/ratio

def recover_labels(predicted):
    print("data_normalization recover: max_ex", max_ex)
    labels=predicted*0.01
    print("max labels: ", np.max(labels))
    return labels

def recover_labels_indi(predicted, max_ex): #N, 128 || N
    print(predicted.shape)
    print("data_normalization recover: max_ex", max_ex)
    print("ratio ", max_ex/default_max_ex)
    labels=((predicted.T)*0.01*max_ex/default_max_ex).T
    print("max labels: ", np.max(labels))
    return labels
    
    