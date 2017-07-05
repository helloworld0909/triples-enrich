import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale



def LR():

    X = []
    Y = []
    valid_split = 0.1
    random.seed(0)

    with open('property.txt', 'r', encoding='utf-8') as input_file:
        for line in input_file:
            name, _, split_count, split_rate, label = line.strip().split('\t')

            split_count = int(split_count)
            split_rate = float(split_rate)
            label = int(label)

            X.append((split_rate, split_count))
            Y.append(label)

    X = scale(X)

    XY = list(zip(X, Y))
    random.shuffle(XY)

    split_index = int(valid_split * len(XY))
    X_train, Y_train = zip(*XY[split_index:])
    X_test, Y_test = zip(*XY[:split_index])


    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    Y_predict = lr.predict(X_test)

    precision = sum(map(lambda y1y2: int(y1y2[0] == y1y2[1]), zip(Y_test, Y_predict))) / float(len(Y_test))
    print(precision)

if __name__ == '__main__':
    LR()