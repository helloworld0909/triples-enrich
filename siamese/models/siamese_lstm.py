from keras.models import Model
from keras.layers import *
from keras import backend as K


class SiameseLSTM(object):
    params = {'wordEmbeddingDim': 100, 'lstmOutDim': 100}

    attrVocabSize = 0
    valueVocabSize = 0

    maxAttrLen = 10
    maxValueLen = 20

    def __init__(self, **kwargs):
        self.vocabSize = kwargs.get('vocabSize', 1000)

    @staticmethod
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        """
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = 1.0
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    @staticmethod
    def contrastive_accuracy(y_true, y_pred):
        margin = 1.0
        return K.mean(K.equal(y_true, K.cast(K.less(y_pred, margin / 2), 'float32')), axis=-1)

    def build(self):
        embedding = Embedding(
            input_dim=self.vocabSize,
            output_dim=self.params['wordEmbeddingDim'],
            trainable=True,
            name='embedding'
        )

        attr_input = Input((self.maxAttrLen,), name='attr_input')
        attr = embedding(attr_input)
        attr_vec = LSTM(self.params['lstmOutDim'], return_sequences=False)(attr)

        value_input = Input((self.maxValueLen,), name='value_input')
        value = embedding(value_input)
        value_vec = LSTM(self.params['lstmOutDim'], return_sequences=False)(value)

        distance = Lambda(self.euclidean_distance)([attr_vec, value_vec])

        model = Model(inputs=[attr_input, value_input], outputs=distance)
        model.compile(optimizer='adam',
                      loss=self.contrastive_loss,
                      metrics=[self.contrastive_accuracy]
                      )
        model.summary()
        return model


if __name__ == '__main__':
    modelWrapper = SiameseLSTM()
    modelWrapper.build()
