import numpy as np
import pickle
from keras.models import load_model
from bootstrap.load_data import init_word2idx, init_sequences, init_raw_sequences, init_labels


attr_word2idx, value_word2idx = init_word2idx()
raw_seq = init_raw_sequences()
all_data = init_sequences('all_data.json', attr_word2idx, value_word2idx)
value_sequences = all_data[1]
value_tag_sequences = all_data[3]

# model = load_model('model.h5')
# y_predict = model.predict(X, verbose=1)
# np.save("pred_label.npy", y_predict[0])
# np.save("pred_seq.npy", y_predict[1])

with open("pred_0.pkl", 'rb') as pklFile:
    y_predict = pickle.load(pklFile)

print(len(y_predict[0]), len(y_predict[1]))

count = 0
punctuation_list = ['/', '、', ',', '，', ';', '；', '|']
for idx, seq in enumerate(y_predict[1]):
    for i, tag in enumerate(seq):
        raw = raw_seq[idx]
        try:
            if np.argmax(tag) == 2 and raw[1][i - len(seq) + len(raw[1])] in punctuation_list:
                print(raw, i, tag)
                count += 1
        except:
            pass

print(count)
