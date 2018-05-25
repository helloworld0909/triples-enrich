from siamese.util.load import init_word2idx, load_labels
from siamese.models.siamese_lstm import SiameseLSTM


def main():
    lookup = init_word2idx()
    x1, x2, y = load_labels('all_data.json', lookup)
    model = SiameseLSTM(vocabSize=len(lookup)).build()
    model.fit([x1, x2], y, validation_split=0.1, epochs=3, shuffle=True)
    model.save('siamese.h5')


if __name__ == '__main__':
    main()

