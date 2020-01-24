import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import random

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def image_spray(y_bins, y_lower, y_upper, e_bins, e_lower, e_upper, num):

    all_imgs = unpickle("cifar-10-batches-py/data_batch_1")
    #src_image = Image.open("turtle.jpg").convert("L")   
    index = random.randint(0, 9999)
    #print(all_imgs.keys())
    channels = np.array(all_imgs["data"][index])
    channels = channels.reshape([3, 32, 32])
    channels = channels.transpose(1, 2, 0)
    print(all_imgs["filenames"][index])
    src_image = Image.fromarray(channels)
    resized_image = src_image.resize((y_bins, e_bins))  
    pix = np.array(resized_image.convert("L"))
    print(pix)
    plt.imshow(src_image)
    plt.show()
    src_sum = np.sum(pix)
    ratio = num/src_sum

    M=sparse.coo_matrix(pix)
    M = M.tocsc()

    energies = np.linspace(e_lower, e_upper, e_bins)
    ys = np.linspace(y_lower, y_upper, y_bins)

    ye = [
            [
                [ys[i], energies[j]] for k in range (int(M[i,j]*ratio)+1)
            ] for i, j in zip(*M.nonzero())
        ]
    ye = np.concatenate([np.array(i) for i in ye])
    #print(ye)
    return ye

if __name__ == "__main__":
    ye = image_spray(50, -29, 30, 50, 0.25, 25, 200000)
    print(ye.shape)
    plt.hist2d(ye[:, 0], ye[:, 1], range=[[-29, 30],[0, 25]], bins=[50, 50])
    plt.show()