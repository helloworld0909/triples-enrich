import logging
from bootstrap.load_data import init_word2idx, init_sequences
from bootstrap.models.multi_task import MultiTaskLSTM


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

attr_word2idx, value_word2idx = init_word2idx()

train_data = init_sequences('iter_0.json', attr_word2idx, value_word2idx)
X_train = list(train_data[:2])
y_train = list(train_data[2:])

test_data = init_sequences('test_data.json', attr_word2idx, value_word2idx)
X_test = list(test_data[:2])
y_test = list(test_data[2:])

modelWrapper = MultiTaskLSTM(attrVocabSize=len(attr_word2idx), valueVocabSize=len(value_word2idx))
model = modelWrapper.build()

model.fit(X_train, y_train, epochs=1, batch_size=64, shuffle=True, validation_data=(X_test, y_test))
model.save('model.h5')
