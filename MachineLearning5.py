import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    N=len(X)
    means = X[np.random.choice(range(N), K, False), :]
    covariances = [np.cov(X.T) for k in range(K)]
    priors = np.full(K, 1/K)
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    max_iterations = 100
    N = X.shape[0]
    responsibilities = np.zeros((N, K))   
    for _ in range(max_iterations):
        for k in range(K):
            responsibilities[:, k] = stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k]) * priors[k]
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)       
        for k in range(K):
            means[k] = np.sum(responsibilities[:, k][:, None] * X, axis=0) / responsibilities[:, k].sum()
            X_centered = X - means[k]
            covariances[k] = (X_centered.T * responsibilities[:, k]) @ X_centered / responsibilities[:, k].sum()  
            priors[k] = responsibilities[:, k].sum() / N
    assignments = np.argmax(responsibilities, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    plt.figure(figsize=(8, 8))
    cluster_colors = ['purple', 'blue', 'green', 'red']
    for k in range(K):
        cluster_data = X[assignments == k]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, color=cluster_colors[k],)
    def plot_contours(mean, cov, color, linestyle):
        x = np.linspace(-8, 8, 100)
        y = np.linspace(-8, 8, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.dstack((X, Y))
        rv = stats.multivariate_normal(mean, cov)
        density = rv.pdf(Z)
        plt.contour(X, Y, density, levels=[0.01], colors=color, linestyles=linestyle)
    for i in range(K):
        plot_contours(group_means[i], group_covariances[i], 'black', 'dashed')
    for i in range(K):
        plot_contours(means[i], covariances[i], cluster_colors[i], 'solid')    
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.show()    
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)