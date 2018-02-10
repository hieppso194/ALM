import numpy as np
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd

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


def estimate_prob(t_0,t_1,f_0,f_1,f,t):
    for i in range(0,len(f)):
        f_0[i] = float((f_0[i]+1)/(f[i]+1))
        f_1[i] = float((f_1[i]+1)/(f[i]+1))
    for i in range(0,t.shape[0]):
        for j in range(0,t.shape[1]):
            t_0[i][j] = float((t_0[i][j]+1)/(t[i][j]+1))
            t_1[i][j] = float((t_1[i][j]+1)/(t[i][j]+1))
    return t_0,t_1,f_0,f_1


# t_i - transition matrix|y=i, f_0 - count(word|y=0), f_1 - count(word|y=1), f_i = count(word_i) t - count transition matrix
def estimate(row,t_0,t_1,f_0,f_1,f,t,n_gram,y):
    bag_word = bag_words[n_gram]
    word_pre = '0'
    for i in range(0,len(row)-n_gram+1):
        word = row[i: i+n_gram]
        indx = bag_word.index(word)
        f[indx+1] +=1
        if word_pre in bag_word:
            index_pre = bag_word.index(word_pre)
        else:
            index_pre = -1
        t[index_pre+1][indx+1] += 1
        if y == 0:
            f_0[indx+1] +=1
            t_0[index_pre+1][indx+1] +=1
        else:
            f_1[indx + 1] += 1
            t_1[index_pre + 1][indx + 1] += 1
    return t_0,t_1,f_0,f_1,f,t


def hmm_predict_row(row,n_gram,t_0,t_1,f_0,f_1):
    prob_0 = 1
    prob_1 = 1
    word_pre = '0'
    bag_word = bag_words[n_gram]

    for i in range(0,len(row)-n_gram+1):
        word = row[i: i+n_gram]
        indx = bag_word.index(word)
        if word_pre in bag_word:
            index_pre = bag_word.index(word_pre)
        else:
            index_pre = -1
        prob_0 = prob_0 * f_0[indx+1] * t_0[index_pre+1][indx+1]
        prob_1 = prob_1 * f_1[indx + 1] * t_1[index_pre + 1][indx+1]

    if prob_0 > prob_1:
        return 0
    else:
        return 1

def hmm_predict(X,n_gram,t_0,t_1,f_0,f_1):
    label = []
    for i in range(0,len(X)):
        row = X[i]
        y = hmm_predict_row(row,n_gram,t_0,t_1,f_0,f_1)
        label.append(y)
    return label

def hmm_train(X,y,n_gram):
    bag_word = bag_words[n_gram]
    t = np.zeros(shape=(len(bag_word) + 1, len(bag_word) + 1))
    t_0 = np.zeros(shape=(len(bag_word)+1,len(bag_word)+1))
    t_1 = np.zeros(shape=(len(bag_word) + 1, len(bag_word) + 1))
    f = [0 for i in range(0, len(bag_word) + 1)]
    f_0 = [0 for i in range(0,len(bag_word)+1)]
    f_1 = [0 for i in range(0, len(bag_word) + 1)]
    idx = 0
    for row in X:
        t_0, t_1, f_0, f_1, f, t = estimate(row, t_0, t_1, f_0, f_1, f, t, n_gram, y[idx])
        idx +=1
            # encode_row = n_gram_encode(row,n_gram)
    t_0, t_1, f_0, f_1 = estimate_prob(t_0,t_1,f_0,f_1,f,t)
    return t_0, t_1, f_0, f_1


def hmm(filename,y,n_gram):
    bag_word = bag_words[n_gram]
    t = np.zeros(shape=(len(bag_word) + 1, len(bag_word) + 1))
    t_0 = np.zeros(shape=(len(bag_word)+1,len(bag_word)+1))
    t_1 = np.zeros(shape=(len(bag_word) + 1, len(bag_word) + 1))
    f = [0 for i in range(0, len(bag_word) + 1)]
    f_0 = [0 for i in range(0,len(bag_word)+1)]
    f_1 = [0 for i in range(0, len(bag_word) + 1)]
    idx = 0
    X =[]
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for index, row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['", "")
            row = row.replace("']", "")
            t_0, t_1, f_0, f_1, f, t = estimate(row, t_0, t_1, f_0, f_1, f, t, n_gram, y[idx])
            idx +=1
            X.append(row)
            # encode_row = n_gram_encode(row,n_gram)
    t_0, t_1, f_0, f_1 = estimate_prob(t_0,t_1,f_0,f_1,f,t)
    return X,t_0, t_1, f_0, f_1

def phi(row,n_gram):
    bag_word = bag_words[n_gram]

    a = np.zeros(len(bag_word))
    array = list()
    for i in range(0,len(row)-n_gram+1):
        i_gram = i+n_gram
        array.append(row[i:i_gram])
        word = row[i:i_gram]
    keys,values = list(Counter(array).keys()),list(Counter(array).values())
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

def n_gram_encode(row,n_gram):
    bag_word = bag_words[n_gram]

    a = []
    for i in range(0, len(row) - n_gram + 1):
        i_gram = i + n_gram
        word = row[i:i_gram]
        pos =  bag_word.index(word)
        a.append(pos)
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
            # encode_row = n_gram_encode(row,n_gram)
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
def read_data(filename):
    data = []
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(readCSV):
            row = str(row)
            row = row.replace("['", "")
            row = row.replace("']", "")
            data.append(row)
    return data

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

# print (count_zero_value([0,0,1],0.8))
def feature_selection(count):
    features_removed = list()
    a_df = pd.DataFrame(count)
    col = []
    for i in range(0,a_df.shape[1]):
        col_i = np.array(a_df.loc[:,i]).mean()
        col.append(col_i)
    for i in range(0,a_df.shape[1]):
        check = 0
        for j in range(0,a_df.shape[0]):
            if a_df.loc[j,i] != col[i]:
                check = 1
                break
        if check ==0:
            features_removed.append(i)
    return features_removed

# ======================================================================================================================
# Test data
# =============================================================
# Train dataset===========================================================
# print(n_gram_encode("TAGATGTCAATTTTAAACTAAGGCTTGATATACTTGGCCTTCAGGTAAAGAGCGGCTATCCCAGTCAAATGTGACTGGCACCCAAGCAGATTATAAGGGAG",2))
n_gram = 5
Y = read_label('../data/Ytr0.csv')
Y1 = read_label('../data/Ytr1.csv')
Y2 = read_label('../data/Ytr2.csv')

Y_all = np.concatenate((Y,Y1,Y2),axis=0)
X,t_0,t_1,f_0,f_1 = hmm('../data/Xtr0.csv',Y,n_gram)
X1,t_0_1,t_1_1,f_0_1,f_1_1 = hmm('../data/Xtr1.csv',Y1,n_gram)
X2,t_0_2,t_1_2,f_0_2,f_1_2 = hmm('../data/Xtr2.csv',Y2,n_gram)
X = read_data('../data/Xtr0.csv')
X1 = read_data('../data/Xtr1.csv')
X2 = read_data('../data/Xtr2.csv')
X_all = np.concatenate((X,X1,X2),axis=0)
t_0_all,t_1_all,f_0_all,f_1_all = hmm_train(X_all,Y_all,n_gram)
# print (X_all.shape)
#
# Xt = hmm('../data/Xte0.csv',n_gram,Y)
# Xt1 = hmm('../data/Xte1.csv',n_gram,Y)
# Xt2 = hmm('../data/Xte2.csv',n_gram,Y)


# ================================================================================================
# label train dataset

#
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
# X_train = np.transpose(X_train)
# X_test = np.transpose(X_test)
#
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.33)
# X_train1 = np.transpose(X_train1)
# X_test1 = np.transpose(X_test1)
#
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.33)
# X_train2 = np.transpose(X_train2)
# X_test2 = np.transpose(X_test2)
# all dataset
# X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all.T, Y_all, test_size=0.33)
# X_train_all = np.transpose(X_train_all)
# X_test_all = np.transpose(X_test_all)

# y_train_all = np.transpose(y_train_all)

# # ====================================================================
#  initialize parameters
# eta = .005
# d = X_train_all.shape[0]
# w_init = np.random.rand(d,1)
#===============================================================================
# w1 = logistic_regression(X1, Y1, w_init, eta)
# wtest_all = logistic_regression(X_train_all,y_train_all,w_init,eta)

# #==============================================================================
# w = logistic_regression(X, Y, w_init, eta)
# w1 = logistic_regression(X1, Y1, w_init, eta)
# w2 = logistic_regression(X2, Y2, w_init, eta)
# wtest_all = w
# y_pred_test_all = hmm_predict(X_test_all,wtest_all)
# w_all = logistic_regression(X_all,Y_all,w_init,eta)
y_pred_all = hmm_predict(X_all,n_gram,t_0_all,t_1_all,f_0_all,f_1_all)
y_pred = hmm_predict(X,n_gram,t_0,t_1,f_0,f_1)
y_pred1 = hmm_predict(X1,n_gram,t_0_1,t_1_1,f_0_1,f_1_1)
y_pred2 = hmm_predict(X2,n_gram,t_0_2,t_1_2,f_0_2,f_1_2)
#============================================================================================================================
# yte = predict(Xt,w)
# yte1 = predict(Xt1,w1)
# yte2 = predict(Xt2,w2)
# thefile = open('../data/Yte.csv', 'w')
# thefile.write("%s\n" % "Id,Bound")
# for index, item in enumerate(yte):
#     thefile.write("%s" % index)
#     thefile.write("%s" % ",")
#     thefile.write("%s\n" % item)
# for index, item in enumerate(yte1):
#     index +=1000
#     thefile.write("%s" % index)
#     thefile.write("%s" % ",")
#     thefile.write("%s\n" % item)
# for index, item in enumerate(yte2):
#     index+=2000
#     thefile.write("%s" % index)
#     thefile.write("%s" % ",")
#     thefile.write("%s\n" % item)
# #
# #
#
# print ("Write completed")
#
print (accuracy_score(y_pred_all,Y_all))
print (accuracy_score(y_pred,Y))
print (accuracy_score(y_pred1,Y1))
print (accuracy_score(y_pred2,Y2))