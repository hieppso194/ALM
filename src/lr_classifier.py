import numpy as np
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
from sklearn import svm

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
# ===============================================================================================
# encode n-grams

DNA_characters = ['A', 'C', 'G', 'T']

def extend_bag_words(bag_words, n_gram=4):
    new = list()

    for i in range(len(bag_words)):
        for j in range(n_gram):
            new.append(bag_words[i]+DNA_characters[j])
    return new

two_gram_bag_words = extend_bag_words(DNA_characters)
three_gram_bag_words = extend_bag_words(two_gram_bag_words)
four_gram_bag_words = extend_bag_words(three_gram_bag_words)
five_gram_bag_words = extend_bag_words(four_gram_bag_words)
six_gram_bag_words = extend_bag_words(five_gram_bag_words)
seven_gram_bag_words = extend_bag_words(six_gram_bag_words)

bag_words = [0,DNA_characters,two_gram_bag_words,three_gram_bag_words,four_gram_bag_words,five_gram_bag_words,six_gram_bag_words,seven_gram_bag_words]

# print (bag_words)

def phi(row,n_gram):
    bag_word = bag_words[n_gram]

    a = np.zeros(len(bag_word))
    array = list()
    for i in range(0,len(row)-n_gram+1):
        i_gram = i+n_gram
        array.append(row[i:i_gram])
    keys,values = Counter(array).keys(),Counter(array).values()
    for i in range(len(keys)):
        for j in range(len(bag_word)):
            if keys[i] == bag_word[j]:
                a[j] = values[i]
    return  a

def phi_combine_gram(row):
    one_gram_phi = phi(row,1)
    two_gram_phi = phi(row,2)
    three_gram_phi = phi(row,3)
    four_gram_phi = phi(row,4)
    five_gram_phi = phi(row,5)

    a = np.concatenate((two_gram_phi,three_gram_phi,four_gram_phi), axis=0)

    return a

def spectrum_encode(filename, n_gram):

    convert_data = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for index, row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['", "")
            row = row.replace("']", "")

            encode_row = phi(row,n_gram)
            convert_data.append(encode_row)
    convert_data = np.asarray(convert_data)
    return convert_data.T

# bag of words with idf
def spectrum_encode_inverse(filename, n_gram):
    m = spectrum_encode(filename, n_gram)
    D = m.shape[0]
    N = m.shape[1]

    documents_contain = np.zeros(D)
    for i in range(D):
        for j in range(N):
            if m[i][j] != 0:
                documents_contain[i]+=1

    # idf = documents_contain/N
    idf = np.log(N/documents_contain)

    for k in range(D):
        m[i,:] *= idf[i]
    return m

# bag of words combines n-gram bagofword
def spectrum_encode_combine_gram(filename):

    convert_data = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for index, row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['", "")
            row = row.replace("']", "")

            encode_row = phi_combine_gram(row)
            convert_data.append(encode_row)
    convert_data = np.asarray(convert_data)
    return convert_data.T

# ============================================================================================
# Read label from Ytrk.csv

def read_label(filename):
    label = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(readCSV):
            if row[1] != 'Bound':
                label.append(int(row[1]))
    label = np.asarray(label)
    return label

# ===============================================================================================
# Common functions

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def tan(s):

    return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))

# ================================================================================================
# Logistic regression

def logistic_regression(X, Y, w_init, eta, tol = 1e-6, max_count = 1e5, lam = 0.01):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 30
    while count < max_count:
        # choose i randomly
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:,i].reshape(d,1)
            yi = Y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            g = (yi-zi)*xi + lam*w[-1]/N
            w_new = w[-1] + eta*g
            count +=1

            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w

            w.append(w_new)
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

# ==============================================================================================================
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
# ===================================================================================================================
# Soft Support Vector Machine

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def SSVM(X,Y,C,bound=1e-7):
    N = X.shape[0]
    D = X.shape[1]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = linear_kernel(X[i], X[j])
            # K[i, j] = polynomial_kernel(X[i], X[j])

    P = np.dot(np.dot(Y,Y.T),K)

    P = matrix(P)
    q = matrix(-np.ones((N, 1)))
    G = matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = matrix(np.vstack((np.zeros((N,1)), C*np.ones((N,1)))))
    A = np.reshape((Y.T), (1,N))
    A = A.astype(float)
    A = matrix(A)

    b = matrix(np.zeros((1,1)))

    solvers.options['show_progress'] =  False
    sol = solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    a = np.ravel(sol['x'])

    # Support vectors have non zero lagrange multipliers
    sv = a > bound
    aS = a[sv]
    xS = X[sv]
    yS = Y[sv]

    w = np.zeros(D)
    for i in range(len(aS)):
        w += aS[i] * yS[i] * xS[i]
    b = yS - np.dot(xS,w)
    b = np.mean(b)
    return w,b

def predict_SSVM(X,w,b):
    product = np.dot(X,w)+b
    print (product.shape)
    return np.sign(product)

# =====================================================================================================================
#  Measures

def accuracy_score(y_pred,y_label):
    count = 0
    N = len(y_pred)
    for i in range(N):
        if y_pred[i] == y_label[i]:
            count += 1

    score = float(count) / N
    return score
# ======================================================================================================================
# Train dataset

X = spectrum_encode('../data/Xtr0.csv',5)
X1 = spectrum_encode('../data/Xtr1.csv',5)
X2 = spectrum_encode('../data/Xtr2.csv',5)

Y = read_label('../data/Ytr0.csv')
Y1 = read_label('../data/Ytr1.csv')
Y2 = read_label('../data/Ytr2.csv')

# =================================================================================================
# Test data set

Xt = spectrum_encode('../data/Xte0.csv',5)
Xt1 = spectrum_encode('../data/Xte1.csv',5)
Xt2 = spectrum_encode('../data/Xte2.csv',5)

# ================================================================================================
# Test accuracy score based on the train dataset

X_train, X_test, y_train, y_test = train_test_split(X.T, Y, test_size=0.33)
X_train = np.transpose(X_train)
X_test = np.transpose(X_test)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1.T, Y1, test_size=0.33)
X_train1 = np.transpose(X_train1)
X_test1 = np.transpose(X_test1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2.T, Y2, test_size=0.33)
X_train2 = np.transpose(X_train2)
X_test2 = np.transpose(X_test2)

# ===================================================================================
#  Initialize parameters

eta = .005
d = X2.shape[0]
w_init = np.random.rand(d,1)
# w_init = 0.1*np.ones((d,1))
#===================================================================================
# Test accuracy score based on the train dataset

# wtest = logistic_regression(X_train,y_train,w_init,eta)
# y_pred_test = predict(X_test,wtest)

# wtest1 = logistic_regression(X_train1,y_train1,w_init,eta)
# y_pred_test1 = predict(X_test1,wtest1)

# wtest2 = logistic_regression(X_train2,y_train2,w_init,eta)
# y_pred_test2 = predict(X_test2,wtest2)

# ==============================================================================
# Learn the test dataset and write to Yte.csv

print ("We are using Logistic Regression.")
print ("Learning the dataset Xtr0.csv.....")
w = logistic_regression(X, Y, w_init, eta)
print ("Learned Xtr0.csv")
print ("Learning the dataset Xtr1.csv.....")
w1 = logistic_regression(X1, Y1, w_init, eta)
print ("Learned Xtr1.csv")
print ("Learning the dataset Xtr2.csv.....")
w2 = logistic_regression(X2, Y2, w_init, eta)
print ("Learned Xtr2.csv")

print ("Predicting the test datasets ....")
yte = predict(Xt,w)
yte1 = predict(Xt1,w1)
yte2 = predict(Xt2,w2)

print ("Writing the results to Yte.csv")
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

print ("Write completed, The result is now in data/Yte.csv.")
# print (accuracy_score(y_pred_test2, y_test2))