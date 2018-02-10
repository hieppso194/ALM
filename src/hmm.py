import numpy as np
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd


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



def accuracy_score(y_pred,y_label):
    count = 0
    N = len(y_pred)
    for i in range(N):
        if y_pred[i] == y_label[i]:
            count += 1

    score = float(count) / N
    return score

# Perceptron


# print (count_zero_value([0,0,1],0.8))

# ======================================================================================================================
# Test data
# =============================================================
# Train dataset===========================================================
# print(n_gram_encode("TAGATGTCAATTTTAAACTAAGGCTTGATATACTTGGCCTTCAGGTAAAGAGCGGCTATCCCAGTCAAATGTGACTGGCACCCAAGCAGATTATAAGGGAG",2))
n_gram = 6
Y = read_label('../data/Ytr0.csv')
Y1 = read_label('../data/Ytr1.csv')
Y2 = read_label('../data/Ytr2.csv')

Y_all = np.concatenate((Y,Y1,Y2),axis=0)
X,t_0,t_1,f_0,f_1 = hmm('../data/Xtr0.csv',Y,n_gram)
X1,t_0_1,t_1_1,f_0_1,f_1_1 = hmm('../data/Xtr1.csv',Y1,n_gram)
X2,t_0_2,t_1_2,f_0_2,f_1_2 = hmm('../data/Xtr2.csv',Y2,n_gram)
Xt = read_data('../data/Xtr0.csv')
Xt1 = read_data('../data/Xtr1.csv')
Xt2 = read_data('../data/Xtr2.csv')
# X_all = np.concatenate((X,X1,X2),axis=0)
# t_0_all,t_1_all,f_0_all,f_1_all = hmm_train(X_all,Y_all,n_gram)
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
# y_pred_all = hmm_predict(X_all,n_gram,t_0_all,t_1_all,f_0_all,f_1_all)
# y_pred = hmm_predict(Xt,n_gram,t_0,t_1,f_0,f_1)
# y_pred1 = hmm_predict(Xt1,n_gram,t_0_1,t_1_1,f_0_1,f_1_1)
# y_pred2 = hmm_predict(Xt2,n_gram,t_0_2,t_1_2,f_0_2,f_1_2)
#============================================================================================================================
# yte = predict(Xt,w)
# yte1 = predict(Xt1,w1)
# yte2 = predict(Xt2,w2)
yte = hmm_predict(Xt,n_gram,t_0,t_1,f_0,f_1)
yte1 = hmm_predict(Xt1,n_gram,t_0_1,t_1_1,f_0_1,f_1_1)
yte2 = hmm_predict(Xt2,n_gram,t_0_2,t_1_2,f_0_2,f_1_2)
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

print ("Write completed")
# #
# #
#
# print ("Write completed")
#
# print (accuracy_score(y_pred_all,Y_all))
# print (accuracy_score(y_pred,Y))
# print (accuracy_score(y_pred1,Y1))
# print (accuracy_score(y_pred2,Y2))