import numpy as np
import pandas as pd



X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 44727 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train=X[0:50000]
    y_train=y[0:50000]
    X_test=X[50000:94727]
    y_test=y[50000:94727]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    K=max(y_train)
    prior_probailities=[np.mean([y==i+1])for i in range (K)]
    class_priors=prior_probailities
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    K = max(y_train)
    D = X.shape[1]
    pAcd = np.zeros((K, D))
    pCcd = np.zeros((K, D))
    pGcd = np.zeros((K, D))
    pTcd = np.zeros((K, D))
    for c in range(K):
        for d in range(D):
            pAcd[c, d] = np.mean((X[:, d][y == c + 1] == 'A'))  
            pTcd[c, d] = np.mean((X[:, d][y == c + 1] == 'T'))
            pGcd[c, d] = np.mean((X[:, d][y == c + 1] == 'G'))
            pCcd[c, d] = np.mean((X[:, d][y == c + 1] == 'C'))
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    K = max(y_train)
    N = X.shape[0]  
    score_values = np.zeros((N, K))
    for k in range(K):
        for n in range(N):
            score = 1.0
            for d in range(X.shape[1]):
                    score*=(pAcd[k][d]**(X[n,d]=='A'))*(pCcd[k][d]**(X[n,d]=='C'))*(pGcd[k][d]**(X[n,d]=='G'))*(pTcd[k][d]**(X[n,d]=='T'))
            score_values[n, k] = np.log(score) + np.log(class_priors[k])
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = scores.shape[1]  # Number of classes
    confusion_matrix = np.zeros((K, K)).astype(int)

    # Assign each data point to the class with the highest score
    predicted_classes = np.argmax(scores, axis=1)

    for i in range(len(y_truth)):
        true_class = y_truth[i] - 1  # Adjust for 0-based indexing
        predicted_class = predicted_classes[i]
        confusion_matrix[predicted_class,true_class, ] += 1  # Use true_class and predicted_class as indices

    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
