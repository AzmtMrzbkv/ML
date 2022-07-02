"""
Naive-Bayes classifier.

Azamat Myrzabekov
Student ID: 20192022
"""

from cmath import log
import numpy as np

def import_data(name):
    print('importing dataset ....')
    D = np.genfromtxt(name, dtype=float, delimiter=',')
    print(f'dataset shape: {D.shape}\n')

    return D

def split_dataset(D):
    # the last column assumed to be categorical value
    X, y = [d[:d.shape[0] - 1] for d in D], [d[d.shape[0] - 1:] for d in D]

    return np.array(X), np.array(y, dtype=int).reshape(D.shape[0])

def divide_dataset(D, alpha):
    print('randomy dividing dataset into train dataset and test dataset ....')
    print(f'train dataset: {round(alpha * 100, 1)}%, test dataset: {round((1 - alpha) * 100, 1)}%')
    np.random.shuffle(D)
    train, test = D[: int(D.shape[0]*alpha)], D[int(D.shape[0]*alpha) + 1:]
    print(f'train dataset shape: {train.shape}, test dataset shape: {test.shape}\n')

    return train, test

def predict(X_test, y, X):
    y_pred = []
    for label in X_test:
        y_pred.append(naive_predict(label, y, X))
    
    return np.array(y_pred)

def naive_predict(l, y, X):
    # count[0] for category value 0 and count[1] for category value 1
    count_l = np.zeros((2, l.shape[0]), dtype=int)
    
    # count positive category value
    count_yes = 0

    # count corresponding attributes
    for i in range(y.shape[0]):
        for j in range(l.shape[0]):
            if X[i][j] == l[j]:
                count_l[y[i]][j] += 1
            count_yes += y[i]

    # compute log probabilities
    P_yes, P_no = log(count_yes / y.shape[0]).real, log((y.shape[0] - count_yes)/y.shape[0]).real
    for i in range(l.shape[0]):
        if count_l[0][j] == 0 or count_l[1][j] == 0:
            continue
        P_yes += log((count_l[1][j]) / count_yes).real
        P_no += log((count_l[0][j]) / (y.shape[0] - count_yes)).real
    
    if P_no > P_yes:
        return 0
    else:
        return 1

def compute_accuracy(y_test, y_pred):
    TP, TN = 0, 0

    for i in range(y_test.shape[0]):
        if y_test[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_test[i] == 0 and y_pred[i] == 0:
            TN += 1
    
    return (TP + TN) / y_test.shape[0]

# there are a lot of data with missing values 
# def remove_missing(X):
#     # remove labels with possibly missing values.
#     good_X = []

#     for i in X:
#         if 0 not in i[:-1]:
#             good_X.append(i)
    
#     print(f'>>> the shape of the array after removing labels with possible missing attributes: ({len(good_X)}, {len(good_X[0])})\n')
#     return np.array(good_X)

def replace_missing(D):
    # treat all '0' as missing value
    att_sum = np.zeros(D.shape[1] - 1)
    att_count = np.zeros(D.shape[1] - 1, dtype=int)

    for d in D:
        for i in range(D.shape[1] - 1):
            if d[i] != 0:
                att_sum[i] += d[i]
                att_count[i] += 1
    
    # compute mean
    att_mean = np.divide(att_sum, att_count)

    # fill missing values
    for i in range(D.shape[0]):
        for j in range(D.shape[1] - 1):
            if D[i][j] == 0:
                D[i][j] = att_mean[j]

    return D

def main():
    # import dataset
    Data = import_data('pima-indians-diabetes.csv')

    # # remove missing values
    # good_Data = replace_missing(Data)    

    # replace the missing values with mean
    good_Data = replace_missing(Data)

    # 80% of dataset => train dataset, 20% of dataset => test dataset
    train_Data, test_Data = divide_dataset(good_Data, 0.8)

    # split categorical value and attributes
    X_train, y_train = split_dataset(train_Data)
    X_test, y_test = split_dataset(test_Data)

    # prediction
    y_pred = predict(X_test, y_train, X_train)

    #compute accuracy
    accuracy = compute_accuracy(y_test, y_pred)
    print(f'test dataset is predicted with {round(accuracy * 100, 2)}% accuracy\n')

if __name__ == '__main__':
    main()