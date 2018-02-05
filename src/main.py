import numpy as np
import csv

def encode_onehot(row):
    DNA = 'ACGT'
    char_to_int = dict((c, i) for i, c in enumerate(DNA))
    int_to_char = dict((i, c) for i, c in enumerate(DNA))

    integer_encode = [char_to_int[char] for char in row]
    onehot_encode = list()
    for value in integer_encode:
        letter = [0 for _ in range(len(DNA))]
        letter[value] = 1
        onehot_encode.append(letter)

    onehot_encode = np.asarray(onehot_encode)
    onehot_encode = np.reshape(onehot_encode, -1)
    return onehot_encode
# encode one-hot
def encode_data(filename):
    convert_data = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for index,row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['","")
            row = row.replace("']","")

            encode_row = encode_onehot(row)
            convert_data.append(encode_row)
    convert_data = np.asarray(convert_data)
    return convert_data.T

# encode numerical
def transfer_number(row):
    DNA = 'ACGT'
    char_to_int = dict((c,i) for i, c in enumerate(DNA))
    integer_encode = [char_to_int[char] for char in row]

    return integer_encode

def encode_numerical(filename):
    convert_data = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for index, row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['", "")
            row = row.replace("']", "")

            encode_row = transfer_number(row)
            convert_data.append(encode_row)
    convert_data = np.asarray(convert_data)
    return convert_data.T

def read_label(filename):
    label = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(readCSV):
            if row[1] != 'Bound':
                label.append(int(row[1]))
    label = np.asarray(label)
    return label

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_regression(X, y, w_init, eta, tol = 1e-4, max_count = 2e5):
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # choose i randomly
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:,i].reshape(d,1)
            yi = Y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count +=1

            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w

            w.append((w_new))

    final_tol = np.linalg.norm(w[-1]-w[-check_w_after])
    # print (final_tol)
    # print (count)
    return w

def predict(X,w):
    pred = sigmoid(np.dot(w[-1].T,X))
    y_pred = list()
    for i in range(pred.shape[1]):
        if pred[0,i] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.asarray(y_pred)

    return y_pred

def accuracy_score(y_pred,y_label):
    count = 0
    N = len(y_pred)
    for i in range(N):
        if y_pred[i] == y_label[i]:
            count += 1

    score = float(count) / N
    return score

# Perceptron
def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


def perceptron(X, y, w_init, max_count=1e3):
    count=0
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    # mis_points = []
    while count<max_count:
        # mix data
        count+=1
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[mix_id[i]]
            if h(w[-1], xi)[0] != yi:  # misclassified point
                # mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break
    return w

def predict_preceptron(w,x):
    pred = h(w[-1],x)
    y_pred = list()
    for i in range(pred.shape[1]):
        if pred[0, i] == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.asarray(y_pred)
    return y_pred
# Test data

X = encode_data('../data/Xtr0.csv')
X1 = encode_data('../data/Xtr1.csv')
X2 = encode_data('../data/Xtr2.csv')

# X = encode_numerical('../data/Xtr0.csv')
# X1 = encode_numerical('../data/Xtr1.csv')
# X2 = encode_numerical('../data/Xtr2.csv')

Xt = encode_data('../data/Xte0.csv')
Xt1 = encode_data('../data/Xte1.csv')
Xt2 = encode_data('../data/Xte2.csv')

# Xt = encode_numerical('../data/Xte0.csv')
# Xt1 = encode_numerical('../data/Xte1.csv')
# Xt2 = encode_numerical('../data/Xte2.csv')

X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
X1 = np.concatenate((np.ones((1, X1.shape[1])), X1), axis=0)
X2 = np.concatenate((np.ones((1, X2.shape[1])), X2), axis=0)

Xt = np.concatenate((np.ones((1, Xt.shape[1])), Xt), axis=0)
Xt1 = np.concatenate((np.ones((1, Xt1.shape[1])), Xt1), axis=0)
Xt2 = np.concatenate((np.ones((1, Xt2.shape[1])), Xt2), axis=0)

Y = read_label('../data/Ytr0.csv')
Y1 = read_label('../data/Ytr1.csv')
Y2 = read_label('../data/Ytr2.csv')

eta = .005
d = X.shape[0]
w_init = np.random.rand(d,1)

w = logistic_regression(X, Y, w_init, eta)
w1 = logistic_regression(X1, Y1, w_init, eta)
w2 = logistic_regression(X2, Y2, w_init, eta)

y_pred = predict(X,w)
y_pred1 = predict(X1,w1)
y_pred2 = predict(X2,w2)

yte = predict(Xt,w)
yte1 = predict(Xt1,w1)
yte2 = predict(Xt2,w2)

thefile = open('../data/Yte.csv', 'w')
thefile.write("%s\n" % "Id,Bound")
for index, item in enumerate(yte):
    thefile.write("%s" % index)
    thefile.write("%s" % ",")
    thefile.write("%s\n" % item)
for index, item in enumerate(yte1):
    index +=1000
    thefile.write("%s" % index)
    thefile.write("%s" % ",")
    thefile.write("%s\n" % item)
for index, item in enumerate(yte2):
    index+=2000
    thefile.write("%s" % index)
    thefile.write("%s" % ",")
    thefile.write("%s\n" % item)

# Perceptron test
# w = perceptron(X,Y,w_init)
# y_predict = predict_preceptron(w,X)
# # print (w[-1])
# # print (predict_preceptron(w,X))
#
# print (accuracy_score(y_predict,Y))
print (accuracy_score(y_pred,Y),accuracy_score(y_pred1,Y1),accuracy_score(y_pred2,Y2))
