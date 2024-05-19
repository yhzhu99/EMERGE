import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, Precision, Recall
from torchmetrics.classification import BinaryF1Score, ConfusionMatrix
import numpy as np
from sklearn import metrics as sklearn_metrics

def minpse(preds, labels):
    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score

def get_binary_metrics(preds, labels):
    accuracy = Accuracy(task="binary", threshold=0.5)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    f1 = BinaryF1Score()

    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    f1(preds, labels)

    minpse_score = minpse(preds, labels) 

    # return a dictionary
    return {
        "accuracy": accuracy.compute().item(),
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "f1": f1.compute().item(),
        "minpse": minpse_score,
    }

def get_all_metrics(y_outcome_pred, y_readmission_pred, y_outcome_true, y_readmission_true):
    outcome_metrics = get_binary_metrics(y_outcome_pred, y_outcome_true)
    readmission_metrics = get_binary_metrics(y_readmission_pred, y_readmission_true)
    # Merging with prefixes
    merged_dict = {f"outcome_{k}": v for k, v in outcome_metrics.items()}
    merged_dict.update({f"readmission_{k}": v for k, v in readmission_metrics.items()})
    return merged_dict

def check_metric_is_better(cur_best, score, main_metric='outcome_auroc'):
    if cur_best == {}:
        return True
    if score > cur_best[main_metric]:
        return True
    return False
