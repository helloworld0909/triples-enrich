import numpy as np
import pickle
import json
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


with open("all_data.json", 'rb') as jsonFile:
    all_data = json.load(jsonFile)

with open("pred_0.pkl", 'rb') as pklFile:
    y_predict = pickle.load(pklFile)

print(len(y_predict[0]), len(y_predict[1]))

count = 0
punctuation_list = ['/', '、', ',', '，', ';', '；', '|']
punctuation_list = list(map(lambda c: value_word2idx[c], punctuation_list))

for idx, pred in enumerate(zip(y_predict[0], y_predict[1])):
    label, seq = pred
    attr, value = raw_seq[idx]
    indices = all_data[attr]['values'][value]
    for i, tag in enumerate(seq):
        try:
            real_idx = i - len(seq) + len(value)
            if np.argmax(label) == 1 and np.argmax(tag) == 2 and value_sequences[idx][i] in punctuation_list:
                indices.append(real_idx)
                print(attr, value, i, tag, label)
                count += 1
        except:
            pass
    all_data[attr]['values'][value] = list(sorted(set(indices)))

print(count, count / len(y_predict[0]))

with open("all_data_iter1.json", 'wb') as jsonFile:
    json.dump(all_data, jsonFile)
