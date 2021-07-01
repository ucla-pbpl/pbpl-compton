import numpy as np
import matplotlib.pyplot as plt

z_weights = (0.0006*np.linspace(0, 127, 128)**2+0.2)/5
default_max_ex = 8e-8
plt.plot(z_weights, label="z_weights")
plt.legend()
plt.show()
train_normalized_coarse = [8, 7, 4, 3.5, 4, 4.5, 6]
train_normalized = np.interp(np.linspace(0, 127, 128), np.linspace(0, 127, 7), train_normalized_coarse)
trian_raw = train_normalized/80*z_weights*default_max_ex

predict_raw_coarse = [0.05, 0.5, 1, 1, 0.6, 0.2, 0.1, 0.07]
predict_raw = np.interp(np.linspace(0, 127, 128), np.linspace(0, 127, 8), predict_raw_coarse)
#max_ex = np.max(predict_raw)
#predict_normalized = predict_raw/max_ex/z_weights*80

examples_weighted = predict_raw/z_weights*80
examples_flattened = examples_weighted
max_weighted_ex = np.max(examples_flattened, axis = 0) # N
print("normalize_examples_indi: normalize_examples_indi: max_weighted_ex", max_weighted_ex)
print("data_normalization normalize: normalize_examples: default_max_ex", default_max_ex)
target = 10*8e-8/default_max_ex
examples_normalized = (examples_flattened)/max_weighted_ex*target #128*64, N
weighted = examples_normalized
print("normalize_examples_indi: normalize_examples_indi: max of weighted",  np.max(weighted))


plt.plot(train_normalized, label="trian_normalized")
plt.plot(weighted, label="predict_normalized")
plt.ylim(bottom=0)
plt.legend()
plt.show()

plt.plot(trian_raw, label="trian_raw")
plt.plot(predict_raw, label='predict_raw')
plt.ylim(bottom=0)
plt.legend()
plt.show()