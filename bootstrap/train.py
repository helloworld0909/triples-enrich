import logging
from bootstrap.load_data import init_word2idx, init_sequences
from bootstrap.models.multi_task import MultiTaskLSTM

from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

attr_word2idx, value_word2idx = init_word2idx()

all_data = init_sequences(attr_word2idx, value_word2idx)
all_data = train_test_split(*all_data, test_size=0.1, random_state=0)
X_train = all_data[:-4:2]
X_test = all_data[1:-4:2]
y_train = all_data[-4::2]
y_test = all_data[-3::2]


modelWrapper = MultiTaskLSTM(attrVocabSize=len(attr_word2idx), valueVocabSize=len(value_word2idx))
model = modelWrapper.build()

model.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=True)
model.save('model.h5')
