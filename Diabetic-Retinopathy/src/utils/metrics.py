"""Wrappers to sklearn/custom metrics used by the codebase
"""

import numpy as np
import scipy.special
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, classification_report,
                             cohen_kappa_score, confusion_matrix, det_curve,
                             f1_score, fbeta_score, hamming_loss, hinge_loss,
                             jaccard_score, log_loss, matthews_corrcoef,
                             plot_det_curve, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, precision_recall_fscore_support)
import matplotlib.pyplot as plt

def quadratic_weighted_kappa_score(gt_labels, pred_labels, weights = 'quadratic'):
    return cohen_kappa_score(gt_labels, pred_labels, weights = weights)

def accuracy_classwise(gt_labels, pred_labels, num_classes=5):
    for i in range(num_classes):
        accuracy = []
        gt_label = gt_labels[gt_labels == i]
        pred_label = pred_labels[gt_labels == i]
        count = (gt_label==pred_label).sum()
        accuracy.append(count/len(gt_label))
    return np.array(accuracy)

def f1_score_classwise(gt_labels, pred_scores, average=None):
    return f1_score(gt_labels, pred_scores, average=average)

def precision_score_classwise(gt_labels, pred_scores, average=None):
    return precision_score(gt_labels, pred_scores, average=average)

def recall_score_classwise(gt_labels, pred_scores, average=None):
    return recall_score(gt_labels, pred_scores, average=average)

def specificity_score_classwise(gt_labels, pred_scores, average=None):
    return recall_score(gt_labels, pred_scores, average=average)

def sensitivity_score_classwise(gt_labels, pred_scores, average=None):
    return recall_score(gt_labels, pred_scores, average=average)

def roc_auc_score_classwise(gt_labels, pred_scores, multi_class="ovr", average=None):
    return roc_auc_score(y_true = gt_labels, y_score = pred_scores, average=average, multi_class=multi_class)

def precision_at_90recall(gt_labels, pred_scores):
    p, r, _ = precision_recall_curve(gt_labels, pred_scores, pos_label=1)
    idx = np.argmin(np.abs(r - 0.9))
    return p[idx]

def recall_at_90precision(gt_labels, pred_scores):
    p, r, _ = precision_recall_curve(gt_labels, pred_scores, pos_label=1)
    idx = np.argmin(np.abs(p - 0.9))
    return r[idx]
