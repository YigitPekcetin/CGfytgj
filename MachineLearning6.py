import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities = np.genfromtxt("hw06_predicted_probabilities.csv", delimiter = ",")

# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated thresholds of size (N + 1,)
def calculate_threholds(predicted_probabilities):
    # your implementation starts below
    sorted_probs = np.sort(predicted_probabilities)
    thresholds = (sorted_probs[:-1] + sorted_probs[1:]) / 2
    thresholds = np.concatenate(([min(predicted_probabilities)/2], thresholds, [0.9836335]))
    # your implementation ends above
    return thresholds

thresholds = calculate_threholds(predicted_probabilities)
print(thresholds)

# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the thresholds of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    # your implementation starts below
    fp_rates = np.zeros_like(thresholds)
    tp_rates = np.zeros_like(thresholds)
  
    for i, threshold in enumerate(thresholds):
        
        predicted_labels = np.where(predicted_probabilities >= threshold,1,-1)
    
        fp = np.sum((predicted_labels == 1) & (true_labels == -1))
        tp = np.sum((predicted_labels == 1) & (true_labels == 1))
        tn = np.sum((predicted_labels == -1) & (true_labels == -1))
        fn = np.sum((predicted_labels == -1) & (true_labels == 1))

        fp_rates[i] = fp / (fp + tn) if (fp + tn) != 0 else 0
        tp_rates[i] = tp / (tp + fn) if (tp + fn) != 0 else 0

    # your implementation ends above
    return fp_rates, tp_rates

fp_rates, tp_rates = calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds)
print(fp_rates)
print(tp_rates)

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates, tp_rates)
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.show()
fig.savefig("hw06_roc_curve.pdf", bbox_inches = "tight")

# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
    # your implementation starts below
    auroc = np.trapz(tp_rates, fp_rates)
    # your implementation ends above
    return auroc

auroc = calculate_auroc(fp_rates, tp_rates)
print("The area under the ROC curve is {}.".format(auroc))