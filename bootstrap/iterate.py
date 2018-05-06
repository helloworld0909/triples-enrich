import numpy as np
from keras.models import load_model
from bootstrap.load_data import init_word2idx, init_sequences


attr_word2idx, value_word2idx = init_word2idx()
all_data = init_sequences('all_data.json', attr_word2idx, value_word2idx)
X = list(all_data[:2])
y = list(all_data[2:])

model = load_model('model.h5')
y_predict = model.predict(X, verbose=1)
np.save("pred_0.npy", y_predict)

print(y_predict.shape)
print(y_predict[0])
