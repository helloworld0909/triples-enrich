from keras.models import Model
from keras.layers import *


class MultiTaskLSTM(object):
    params = {'wordEmbeddingDim': 100, 'lstmOutDim': 100}

    attrVocabSize = 0
    valueVocabSize = 0

    maxAttrLen = 10
    maxValueLen = 30

    attrLabelDim = 2
    valueTaggingDim = 3  # 0 is padding

    def __init__(self, **kwargs):
        self.attrVocabSize = kwargs.get('attrVocabSize', 1000)
        self.valueVocabSize = kwargs.get('valueVocabSize', 1000)

    def build(self):
        attr_input = Input((self.maxAttrLen,), name='attr_input')
        attr = Embedding(
            input_dim=self.attrVocabSize,
            output_dim=self.params['wordEmbeddingDim'],
            input_length=self.maxAttrLen,
            trainable=True,
            name='attr_embedding'
        )(attr_input)
        attr_vec = LSTM(self.params['lstmOutDim'], return_sequences=False)(attr)

        attr_vec_repeat = RepeatVector(self.maxValueLen)(attr_vec)

        value_input = Input((self.maxValueLen,), name='value_input')
        value = Embedding(
            input_dim=self.valueVocabSize,
            output_dim=self.params['wordEmbeddingDim'],
            input_length=self.maxValueLen,
            trainable=True,
            name='value_embedding'
        )(value_input)
        value = Bidirectional(LSTM(self.params['lstmOutDim'], return_sequences=True), name='value_bilstm')(value)
        value_hidden = TimeDistributed(Dense(self.params['lstmOutDim'], activation='relu'), name='hidden_1')(value)

        value_maxpool = GlobalMaxPooling1D()(value)
        task1_merge = concatenate([attr_vec, value_maxpool])
        task1_output = Dense(self.attrLabelDim, activation='softmax', name='task1_output')(task1_merge)

        task2_merge = concatenate([value_hidden, attr_vec_repeat])
        task2_output = TimeDistributed(
            Dense(self.valueTaggingDim, activation='softmax'), name='task2_output')(task2_merge)

        model = Model(inputs=[attr_input, value_input], outputs=[task1_output, task2_output])
        model.compile(optimizer='adam',
                      loss={'task1_output': 'categorical_crossentropy',
                            'task2_output': 'sparse_categorical_crossentropy'},
                      loss_weights={'task1_output': 1, 'task2_output': 1},
                      metrics=['accuracy']
                      )
        model.summary()
        return model


if __name__ == '__main__':
    modelWrapper = MultiTaskLSTM()
    modelWrapper.build()
