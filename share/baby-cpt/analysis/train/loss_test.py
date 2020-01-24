import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

k = tf.keras.losses.KLDivergence()
loss = k([.4, .9, .2], [.5, .8, .12])
print('Loss: ', loss.numpy())

offset = tf.Variable([0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
truth = tf.Variable([0., 1., 2., 5., 6., 4., 0., 3., 0., 5.])
truth_arr = truth.numpy()
data1 = tf.Variable([0., 1., 3., 4., 6., 4., 0., 0., 3., 5.])
data2 = tf.Variable([0, 1, 1, 4, 7, 4, 0, 3, 0, 2])
data3 = data2*2
data4 = tf.Variable([1, 2, 3, 3, 3, 3, 3, 3, 2, 3])

data=[data1, data2, data3, data4]
plt.plot(np.linspace(0, 9, 10), truth_arr, label = "truth")

kl = tf.keras.losses.KLDivergence()
mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()
for d in data:
    d_arr = d.numpy()
    my_kl_loss = 0
    for i in range(len(d_arr)):
        if (d_arr[i]!=0 and truth_arr[i]!=0):
            my_kl_loss += (truth_arr[i]+0.1)*np.log(truth_arr[i]+0.1)/np.log(d_arr[i]+0.1)
    kl_loss = kl(tf.cast(truth, tf.float32)+offset, tf.cast(d, tf.float32)+offset)
    mse_loss = mse(tf.cast(truth, tf.float32), tf.cast(d, tf.float32))
    cce_loss = cce(tf.cast(truth, tf.float32), tf.cast(d, tf.float32))
    plt.plot(np.linspace(0, 9, 10), d.numpy(), 
        label="my_kld:{:.3f} kld:{:.3f}, mse:{:.3f}, cce:{:.3f}".format(
        my_kl_loss, kl_loss.numpy(), mse_loss.numpy(), cce_loss.numpy()))

plt.legend()
plt.savefig("loss_test.png")

