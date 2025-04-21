import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# Your confusion matrix from sklearn:
#       Predicted
#         0      1
#    0 [ TN=7706, FP=289 ]
#    1 [ FN=  38, TP=7957 ]
#cm = np.array([[4414,   28],
# [4394,   48]]
#)
#cm = np.array([[7706, 289],
#               [  38, 7957]])
cm = np.array([[7939,   56],
 [  50, 7945]])
tn, fp, fn, tp = cm.ravel()

# Total samples
total = tn + fp + fn + tp

# Basic rates
accuracy = (tp + tn) / total
precision = tp / (tp + fp)            # a.k.a. Positive Predictive Value
recall = tp / (tp + fn)               # a.k.a. Sensitivity, True Positive Rate
specificity = tn / (tn + fp)          # True Negative Rate
npv = tn / (tn + fn)                  # Negative Predictive Value
f1 = 2 * precision * recall / (precision + recall)

# Additional useful metrics
fpr = fp / (fp + tn)                  # False Positive Rate
fnr = fn / (fn + tp)                  # False Negative Rate
prevalence = (tp + fn) / total
balanced_accuracy = (recall + specificity) / 2
mcc = matthews_corrcoef(
    y_true=[0]* (tn+fp) + [1]* (fn+tp),
    y_pred=[0]* tn       + [1]* fp  + [0]* fn      + [1]* tp
)
# Note: For AUC we’d need scores; here we just illustrate via labels (not meaningful without scores):
try:
    auc = roc_auc_score(
        y_true=[0]* (tn+fp) + [1]* (fn+tp),
        y_score=[0]* tn    + [1]* fp   + [0]* fn   + [1]* tp
    )
except ValueError:
    auc = None  # can't compute proper AUC without continuous scores

# Print results
metrics = {
    "Total samples": total,
    "Accuracy": accuracy,
    "Precision (PPV)": precision,
    "Recall (Sensitivity, TPR)": recall,
    "Specificity (TNR)": specificity,
    "Negative Predictive Value (NPV)": npv,
    "F1 Score": f1,
    "False Positive Rate (FPR)": fpr,
    "False Negative Rate (FNR)": fnr,
    "Prevalence": prevalence,
    "Balanced Accuracy": balanced_accuracy,
    "Matthews CorrCoef": mcc,
    "AUC (label-only stub)": auc,
}

for name, val in metrics.items():
    print(f"{name:<35}: {val:.4f}" if isinstance(val, float) else f"{name:<35}: {val}")
