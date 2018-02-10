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

# -----------------------------------------------------------------------------------------
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

# ========================================================================================
# 2-3 gram encode
def encode_gram(n_gram,row):
    encode_squence = list()
    for idex in range(0,len(row)-n_gram+1):
        i = idex
        i_gram = i+n_gram
        sub_sequence = row[i:i_gram]
        encode_squence.append(encode_onehot(sub_sequence))
    encode_squence = np.asarray(encode_squence)
    sum_element = encode_squence.shape[0]*encode_squence.shape[1]
    encode_squence = np.reshape(encode_squence,sum_element)
    return encode_squence

# print (encode_gram(2,'ACAGT'))

def gram_encode_data(filename,n_gram):
    convert_data = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for index,row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['","")
            row = row.replace("']","")

            encode_row = encode_gram(n_gram,row)
            convert_data.append(encode_row)
    convert_data = np.asarray(convert_data)
    return convert_data.T
# ===============================================================================================
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

    a = np.concatenate((three_gram_phi,four_gram_phi), axis=0)
    # a  = np.concatenate((four_gram_phi,five_gram_phi),axis=0)

    return a

# print (phi_combine_gram('AAGATGGCGCCGGGAGGGTCGAAATTAATGTCAAGGGGCCCGGGCGCCTTGTTAAGTAGATGTCATGAGGTAAATTTCAATAAAACGCCGCCAAAGGGCAA').shape)

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
def read_label(filename):
    label = list()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(readCSV):
            if row[1] != 'Bound':
                label.append(int(row[1]))
    label = np.asarray(label)
    return label

# ------------------------------------------------------------------------------------------


def sigmoid(s):
    return 1/(1 + np.exp(-s))

def tan(s):

    return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))

def logistic_regression(X, Y, w_init, eta, tol = 1e-4, max_count = 1e5, lam = 0.01):
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
            # zi = tan(np.dot(w[-1].T, xi))
            g = (yi-zi)*xi + lam*w[-1]/N
            w_new = w[-1] + eta*g
            count +=1

            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w

            w.append(w_new)

    final_tol = np.linalg.norm(w[-1]-w[-check_w_after])
    # print (final_tol)
    # print (count)
    return w

def predict(X,w):
    pred = sigmoid(np.dot(w[-1].T,X))
    # pred = tan(np.dot(w[-1].T,X))
    y_pred = list()
    for i in range(pred.shape[1]):
        if pred[0,i] > 0.5:
        # if pred[0, i] > 0.0:
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
# ===================================================================================================================
# Soft Support Vector Machine
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

# def gaussian_kernel(x, y, sigma=5.0):
#     return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def SSVM(X,Y,C,bound=1e-7):
    N = X.shape[0]
    D = X.shape[1]
    # K = np.dot(X,X.T)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # K[i, j] = linear_kernel(X[i], X[j])
            K[i, j] = polynomial_kernel(X[i], X[j])

    P = np.dot(np.dot(Y,Y.T),K)
    print(P.shape)
    P = matrix(P)
    # P = matrix(V.T.dot(V))

    # P = matrix(np.dot(X, X.T))
    q = matrix(-np.ones((N, 1)))
    G = matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = matrix(np.vstack((np.zeros((N,1)), C*np.ones((N,1)))))
    A = np.reshape((Y.T), (1,N))
    # A = Y
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

    # print (aS)

    w = np.zeros(D)
    for i in range(len(aS)):
        w += aS[i] * yS[i] * xS[i]
    b = yS - np.dot(xS,w)
    b = np.mean(b)
    print (b)
    print (w)

    return w,b

def predict_SSVM(X,w,b):
    product = np.dot(X,w)+b
    print (product.shape)
    return np.sign(product)

# =====================================================================================================================
#  Feature selection
def count_zero_value(X, threshold):
    count = 0
    for i in X:
        if i == 0:
            count = count + 1
    ratio = float(count)/float(len(X))

    if ratio < threshold:
        return 0
    else :
        return 1

print (count_zero_value([0,0,1],0.8))
def feature_selection(X,threshold):
    D, N = X.shape
    features_removed = list()
    for i in range(D):
        if count_zero_value(X[i,:],threshold) == 1:
            features_removed.append(i)

    X = np.delete(X,features_removed,0)
    return X,features_removed
#
# a = np.zeros((3,2))
# a[1,1] = 1
# a[2,1] = 1
# print (a.shape)
# print (feature_selection(a, 0.9))


# ======================================================================================================================
# Test data
# =============================================================
# Train dataset===========================================================

X = spectrum_encode('../data/Xtr0.csv',5)
X1 = spectrum_encode('../data/Xtr1.csv',5)
X2 = spectrum_encode('../data/Xtr2.csv',5)
#

# Xa,i = feature_selection(X,0.9)
# print (i)

# X = spectrum_encode_inverse('../data/Xtr0.csv',4)
# X1 = spectrum_encode_inverse('../data/Xtr1.csv',4)
# X2 = spectrum_encode_inverse('../data/Xtr2.csv',4)

# print (X.shape)
# print(spectrum_encode_inverse('../data/Xtr0.csv',4))
X_all = np.concatenate((X,X1,X2),axis=1)
# print (X_all.shape)

# combine gram
# X = spectrum_encode_combine_gram('../data/Xtr0.csv')
# X1 = spectrum_encode_combine_gram('../data/Xtr1.csv')
# X2 = spectrum_encode_combine_gram('../data/Xtr2.csv')

# #
# X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
# X1 = np.concatenate((np.ones((1, X1.shape[1])), X1), axis=0)
# X2 = np.concatenate((np.ones((1, X2.shape[1])), X2), axis=0)
# X_all = np.concatenate((np.ones((1, X_all.shape[1])), X_all), axis=0)

# =================================================================================================
# Test data set
# Xt = spectrum_encode('../data/Xte0.csv',2)
# Xt1 = spectrum_encode('../data/Xte1.csv',2)
# Xt2 = spectrum_encode('../data/Xte2.csv',2)

# print (Xt)
#
Xt = spectrum_encode('../data/Xte0.csv',5)
Xt1 = spectrum_encode('../data/Xte1.csv',5)
Xt2 = spectrum_encode('../data/Xte2.csv',5)

# Xt = np.concatenate((np.ones((1, Xt.shape[1])), Xt), axis=0)
# Xt1 = np.concatenate((np.ones((1, Xt1.shape[1])), Xt1), axis=0)
# Xt2 = np.concatenate((np.ones((1, Xt2.shape[1])), Xt2), axis=0)

# ================================================================================================
# label train dataset
Y = read_label('../data/Ytr0.csv')
Y1 = read_label('../data/Ytr1.csv')
Y2 = read_label('../data/Ytr2.csv')

Y_all = np.concatenate((Y,Y1,Y2),axis=0)
#
# --------------------------------------------------------------------
# Test data from train
# X_train, X_test, y_train, y_test = train_test_split(X.T, Y, test_size=0.33)
# X_train = np.transpose(X_train)
# X_test = np.transpose(X_test)
#
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X1.T, Y1, test_size=0.33)
# X_train1 = np.transpose(X_train1)
# X_test1 = np.transpose(X_test1)
#
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X2.T, Y2, test_size=0.33)
# X_train2 = np.transpose(X_train2)
# X_test2 = np.transpose(X_test2)

# all dataset
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all.T, Y_all, test_size=0.33)
X_train_all = np.transpose(X_train_all)
X_test_all = np.transpose(X_test_all)

y_train_all = np.transpose(y_train_all)
# def convert(y):
#     for i, v in enumerate(y):
#         if v == 0:
#             y[i] = -1
#     return y
#
# def de_convert(y):
#     for i, v in enumerate(y):
#         if v == -1:
#             y[i] = 0
#     return y
#
# y_train_all = convert(y_train_all)
# w,b = SSVM(X_train_all,y_train_all,1000)
# y_pred_SSVM = predict_SSVM(X_test_all,w,b)
# y_pred_SSVM = de_convert(y_pred_SSVM)
# print (y_pred_SSVM.shape,y_test_all.shape)

# clf = svm.SVC(kernel='linear', C=100)
# clf.fit(X_train_all,y_train_all)
# y_pred_SSVM = clf.fit(X_test_all)
# print (y_pred_SSVM.shape)

# # ====================================================================
#  initialize parameters
eta = .005
d = X_train_all.shape[0]
w_init = np.random.rand(d,1)
# w_init = 0.1*np.ones((d,1))
#===============================================================================
# wtest = logistic_regression(X_train,y_train,w_init,eta)
# y_pred_test = predict(X_test,wtest)
#
# wtest1 = logistic_regression(X_train1,y_train1,w_init,eta)
# y_pred_test1 = predict(X_test1,wtest1)
#
# wtest2 = logistic_regression(X_train2,y_train2,w_init,eta)
# y_pred_test2 = predict(X_test2,wtest2)
w1 = logistic_regression(X1, Y1, w_init, eta)
wtest_all = logistic_regression(X_train_all,y_train_all,w_init,eta)

# #==============================================================================
w = logistic_regression(X, Y, w_init, eta)
w1 = logistic_regression(X1, Y1, w_init, eta)
w2 = logistic_regression(X2, Y2, w_init, eta)
# wtest_all = w
y_pred_test_all = predict(X_test_all,wtest_all)
w_all = logistic_regression(X_all,Y_all,w_init,eta)
#============================================================================================================================
# y_pred = predict(X,w)
# y_pred1 = predict(X1,w1)
# y_pred2 = predict(X2,w2)
# #
yte = predict(Xt,w)
yte1 = predict(Xt1,w1)
yte2 = predict(Xt2,w2)

#
# yte_all = predict(Xt,w_all)
# yte1_all = predict(Xt1,w_all)
# yte2_all = predict(Xt2,w_all)

# #
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
#
# write to Yte when combine all datasets to X_all

# thefile = open('../data/Yte.csv', 'w')
# thefile.write("%s\n" % "Id,Bound")
# for index, item in enumerate(yte_all):
#     thefile.write("%s" % index)
#     thefile.write("%s" % ",")
#     thefile.write("%s\n" % item)
# for index, item in enumerate(yte1_all):
#     index +=1000
#     thefile.write("%s" % index)
#     thefile.write("%s" % ",")
#     thefile.write("%s\n" % item)
# for index, item in enumerate(yte2_all):
#     index+=2000
#     thefile.write("%s" % index)
#     thefile.write("%s" % ",")
#     thefile.write("%s\n" % item)

print ("Write completed")
# # # print (encode_onehot('ACAG'))
# # #
# # # # Perceptron test
# # # # w = perceptron(X,Y,w_init)
# # # # y_predict = predict_preceptron(w,X)
# # # # # print (w[-1])
# # # # # print (predict_preceptron(w,X))
# # # #
# # # # print (accuracy_score(y_predict,Y))
# print (accuracy_score(y_pred,Y),accuracy_score(y_pred1,Y1),accuracy_score(y_pred2,Y2))
#
# print (accuracy_score(y_pred_test, y_test), accuracy_score(y_pred_test1, y_test1), accuracy_score(y_pred_test2, y_test2))
print (accuracy_score(y_pred_test_all,y_test_all))
# print (accuracy_score(y_pred_SSVM,y_test_all.T))