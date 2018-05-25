import numpy as np
from siamese.util.load import init_word2idx, load_test
from siamese.models.siamese_lstm import SiameseLSTM


def main():
    lookup = init_word2idx()

    model = SiameseLSTM(vocabSize=len(lookup)).build()
    model.load_weights('siamese.h5')

    x1, x2, y = load_test('test_labeled_with_attribute_1.txt', lookup)
    distances = model.predict([x1, x2])
    print(np.size(distances))

    correct = 0
    for dist, label in zip(distances, y):
        if int(dist < 0.8) == int(label):
            correct += 1
    print(correct / float(np.size(distances)))


if __name__ == '__main__':
    main()
