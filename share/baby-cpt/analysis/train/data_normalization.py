import numpy as np

def normalize_examples(examples, max_ex=4e-9):
    z_weights = (0.0006*np.linspace(0, 127, 128)**2+0.2)/5 #0-1-2
    #max_ex = 0.1e-9
    normalized = examples/max_ex/z_weights
    center = np.max(normalized)/2
    return (normalized-center, max_ex)

def normalize_labels(labels):

    ratio = 1 #1.5e7

    return labels/ratio

def recover_labels(predicted, max_ex=1):
    return predicted*1/4e-9*max_ex
    