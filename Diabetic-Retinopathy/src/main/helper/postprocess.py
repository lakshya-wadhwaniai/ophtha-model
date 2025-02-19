import os
from collections import defaultdict
#from turtle import shape

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

import dill as pickle
import src.utils.metrics as metrics_module
import src.viz.plots as viz_eval_module


def log_to_wandb(figures_dict, phase, train_metrics=None, train_loss=None, val_metrics=None, val_loss=None,
                 test_metrics=None, test_loss=None, epochID=0):
    if phase not in ["train", "inference"]:
        raise Exception(
            "Invalid phase! Please choose one of ['train', 'inference']")
    if figures_dict is not None:
        new_fig_dict = convert_to_wandb_images(figures_dict)
        wandb.log(new_fig_dict, step=epochID)
    if phase == "train":
        wandb.log(train_metrics, step=epochID)
        wandb.log(val_metrics, step=epochID)
        wandb.log({"train_loss": train_loss}, step=epochID)
        wandb.log({"val_loss": val_loss}, step=epochID)
    else:
        wandb.log(test_metrics, step=epochID)
        wandb.log({"test_loss": test_loss}, step=epochID)

    return None

def evaluate_classif(gt, preds, metric_type, cfg, phase):
    """ Support only for classification metrics in this function. """
    return_dict = {}
    if f"{metric_type}_metrics" in cfg["eval"]["logging_metrics"]:
        for metric in cfg["eval"]["logging_metrics"]["{}_metrics".format(metric_type)]:
            callable_metric = getattr(metrics_module, metric)
            # print(gt,preds)
            return_dict["{}_{}".format(phase, metric)] = callable_metric(gt, preds)
        return return_dict

def calculate_metrics(cfg, gt_dict, pred_dict, phase, dataloader):
    metrics_dict = {}
    if (cfg["task_type"] == "binary-classification"):
        # print(gt_dict['gt_labels'])
        
        gt_labels = gt_dict['gt_labels']
        
        pred_scores = pred_dict['pred_scores']
        pred_labels = pred_dict['pred_labels']
        
        metrics_dict.update(evaluate_classif(gt_labels, pred_scores, "score", cfg, phase))
        metrics_dict.update(evaluate_classif(gt_labels, pred_labels, "label", cfg, phase))
        metrics_dict.update(evaluate_classif(gt_labels, pred_labels, "cumulative", cfg, phase))        
        
        #add parameter for plotting confusion matrix
        conf_mat = np.zeros((max(gt_labels+1), max(gt_labels+1)), dtype=int)
        for i, p in enumerate(pred_labels):
            conf_mat[int(gt_labels[i])][int(p.item())] += 1
        conf_mat = pd.DataFrame(conf_mat)

        metrics_dict.update(
            {phase + "_confusion_matrix": wandb.Table(dataframe=conf_mat)}
        )



    elif cfg["task_type"] == "multiclass-classification":
        name_to_code_mapping = dataloader.dataset.datasets[0].class_names
        code_to_name_mapping = {v: k for k, v in name_to_code_mapping.items()}
        gt_labels = gt_dict['gt_labels']
        pred_labels = pred_dict['pred_labels']
        pred_scores = pred_dict['pred_scores']

        soft = nn.Softmax(dim=1)
        pred_score_soft = soft(pred_scores)

        fd = evaluate_classif(gt_labels, pred_score_soft, "multiclass_score", cfg, phase)
        fd.update(evaluate_classif(gt_labels, pred_labels, "multiclass_label", cfg, phase))

        df_dict = {}
        for i in range(max(gt_labels+1)):
            feature_name = code_to_name_mapping[i]
            gt_label = gt_labels[gt_labels == i]
            pred_label = pred_labels[gt_labels == i]
            pred_score = pred_score_soft[gt_labels == i]
     
            feature_dict = evaluate_classif(
                gt_label, pred_score, "score", cfg, phase
            )
            feature_dict.update(
                evaluate_classif(gt_label, pred_label, "label", cfg, phase)
            )
            feature_dict.update({f"{k}": v[i] for k, v in fd.items()})

            df_dict[feature_name] = feature_dict
            feature_dict = {f"{feature_name}_{k}": v for k, v in feature_dict.items()}
            metrics_dict.update(feature_dict)

        metrics_dict.update(evaluate_classif(gt_labels, pred_labels, "cumulative", cfg, phase))
        metrics_df = pd.DataFrame.from_dict(df_dict).T
        metrics_df.reset_index(inplace=True)
        metrics_dict.update(
            {phase + "_summary_table": wandb.Table(dataframe=metrics_df)}
        )
        
        #add parameter for plotting confusion matrix
        conf_mat = np.zeros((max(gt_labels+1), max(gt_labels+1)), dtype=int)
        for i, p in enumerate(pred_labels):
            conf_mat[int(gt_labels[i])][int(p.item())] += 1
        conf_mat = pd.DataFrame(conf_mat)

        metrics_dict.update(
            {phase + "_confusion_matrix": wandb.Table(dataframe=conf_mat)}
        )



    elif cfg["task_type"] == "multilabel-classification":
        gt_labels = gt_dict['gt_labels']
        pred_labels = pred_dict['pred_labels']
        pred_scores = pred_dict['pred_scores']
        name_to_code_mapping = dataloader.dataset.datasets[0].default_pathologies
        code_to_name_mapping = {v: k for k, v in name_to_code_mapping.items()}
        df_dict = {}
        uncertain_labels = cfg['data']['ignore_labels']
        uncertain_labels = np.array(uncertain_labels)
        for i in range(gt_labels.shape[1]):
            feature_name = code_to_name_mapping[i]
            gt_label = gt_labels[:,i]
            ignore_idx = np.where(np.isin(gt_label,uncertain_labels))
            gt_label[ignore_idx] = np.nan
            gt_label = gt_label[~np.isnan(gt_label)]
            # If all elements are nan, then particular label doesn't exist in the dataset, 
            # so we can ignore following steps.
            if len(gt_label) > 0:
                pred_label = pred_labels[:,i]
                pred_label[ignore_idx] = np.nan
                pred_label = pred_label[~np.isnan(pred_label)]
                pred_score = pred_scores[:,i]
                pred_score[ignore_idx] = np.nan
                pred_score = pred_score[~np.isnan(pred_score)]
                feature_dict = evaluate_classif(
                    gt_label, pred_score, "score", cfg, phase
                )
                feature_dict.update(
                    evaluate_classif(gt_label, pred_label, "label", cfg, phase)
                )
                df_dict[feature_name] = feature_dict
                feature_dict = {f"{feature_name}_{k}": v for k, v in feature_dict.items()}
                metrics_dict.update(feature_dict)

        metrics_df = pd.DataFrame.from_dict(df_dict).T
        metrics_df.reset_index(inplace=True)
        metrics_dict.update(
            {phase + "_summary_table": wandb.Table(dataframe=metrics_df)}
        )

        
    else:
        raise ValueError(
            "Support for given task_type is not yet present in the metrics module."
            + "Please choose one of - `binary-classification`, `multiclass-classification`,"
            + " or `multilabel-classification`"
        )

    averaging_metrics = cfg["eval"]["averaging_metrics"]
    averaging = cfg["eval"]["averaging"]
    metrics_dict = calculate_mean_metrics(metrics_dict, phase, averaging_metrics, averaging)

    return metrics_dict


def calculate_mean_metrics(metrics_dict, phase, metric_names_list, average="simple"):
    if average == "simple":
        for metric_name in metric_names_list:
            metric_value_list = []
            for metric, value in metrics_dict.items():
                if metric_name in metric:
                    metric_value_list.append(value)
            metric_mean = np.mean(metric_value_list)
            metrics_dict[f"mean_{phase}_{metric_name}"] = metric_mean

        return metrics_dict

        
    else:
        raise(NotImplementedError)

#discuss later
def create_figures(cfg, phase, train_gt_labels=None, train_scores=None, val_gt_labels=None, val_scores=None, 
                   test_gt_labels=None, test_scores=None):
    """
    Plot figures to be logged to wandb.

    phase = "train" plots both the training and validation figures to allow better analysis. Phase = "inference" may be used for either standalone inference on validation set or on test set. The variable names use "test*" format since this will be create plots for a single forward pass (no training).

    Args:
        cfg (dict): Configuration file used to run the code.
        phase (str): Run a training loop or inference loop. Should be one of "train" or "inference".
        train_gt_labels (list, optional): Ground truth labels from the train set. Defaults to None.
        train_scores (list, optional): Class wise scores predicted by the model for the training set. Defaults to None.
        val_gt_labels (list, optional): Ground truth labels from the validation set. Defaults to None.
        val_scores (list, optional): Class wise scores predicted by the model for the validation set. Defaults to None.
        test_gt_labels (list, optional): Ground truth labels from the test set. Defaults to None.
        test_scores (list, optional): Class wise scores predicted by the model for the test set. Defaults to None.

    Raises:
        Exception: Incorrect phase values like "val" or "test" may lead to incorrect performance. Also ensures that case sensitive phase flags are passed.

    Returns:
        dict: A dictionary containing mappings to all figures that need to be plotted according to the config file
    """
    if phase not in ["train", "inference"]:
        raise Exception("Invalid phase! Please choose one of ['train', 'inference']")
    figures_dict = {}
    if phase == "train":
        if cfg["task_type"] == "binary-classification":
            y_true_arr = [train_gt_labels, val_gt_labels]
            y_pred_proba_arr = [train_scores, val_scores]
            labels_arr = ["Train", "Val"]
            plot_thres_for_idx = [0]


            figures_dict = add_figures_to_dict(cfg, phase, figures_dict, train_gt_labels, train_scores, "Train", plot_thres_for_idx)
            figures_dict = add_figures_to_dict(cfg, "val", figures_dict, val_gt_labels, val_scores, "Val", plot_thres_for_idx)

        elif cfg["task_type"] == "multilabel-classification":
            uncertain_labels = cfg['data']['ignore_labels']
            uncertain_labels = np.array(uncertain_labels)
            y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx = populate_multilabel_arrays(train_gt_labels, train_scores, labels_dict, uncertain_labels)
            figures_dict = add_figures_to_dict(cfg, phase, figures_dict, y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx)
            y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx = populate_multilabel_arrays(val_gt_labels, val_scores, labels_dict, uncertain_labels)
            figures_dict = add_figures_to_dict(cfg, "val", figures_dict, y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx)

        elif cfg["task_type"] == "multiclass-classification":

            plot_thres_for_idx = [i for i in range(cfg['model']['params']['num_classes'])]


            figures_dict = add_figures_to_dict(cfg, phase, figures_dict, train_gt_labels, train_scores, [0,1,2,3,4], plot_thres_for_idx)
            figures_dict = add_figures_to_dict(cfg, "val", figures_dict, val_gt_labels, val_scores, [0,1,2,3,4], plot_thres_for_idx)
    else:
        if cfg["task_type"] == "binary-classification":
            y_true_arr = [test_gt_labels]
            y_pred_proba_arr = [test_scores]
            labels_arr = ["Test"]
            plot_thres_for_idx = [0]
            
        elif cfg["task_type"] == "multilabel-classification":
            uncertain_labels = cfg['data']['ignore_labels']
            uncertain_labels = np.array(uncertain_labels)
            y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx = populate_multilabel_arrays(test_gt_labels, test_scores, labels_dict, uncertain_labels)
            figures_dict = add_figures_to_dict(cfg, phase, figures_dict, y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx)
        elif cfg["task_type"] == "multiclass-classification":

            plot_thres_for_idx = [i for i in range(cfg['model']['params']['num_classes'])]

            figures_dict = add_figures_to_dict(cfg, phase, figures_dict, test_gt_labels, test_scores, [0,1,2,3,4], plot_thres_for_idx)
    return figures_dict


def populate_multilabel_arrays(gt_labels, scores, labels_dict, uncertain_labels=[]):
    y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx = [[] for _ in range(4)]
    
     #Using two variables idx and i to iterate as some labels may be nan for all samples
     #idx increases only when a given label has non nan values,otherwise continue block executes
     #plot_thres_for_idx stores index values only when a given label has at least one non nan value

    idx = 0
    for i in range(max(gt_labels+1)):
        gt_label = gt_labels[gt_labels == i]
        # ignore_idx = np.where(np.isin(gt_label,uncertain_labels))
        # gt_label[ignore_idx] = np.nan
        # gt_label = gt_label[~np.isnan(gt_label)]
        pred_score = scores[gt_labels == i][:,i]
        # pred_score[ignore_idx] = np.nan
        # pred_score = pred_score[~np.isnan(pred_score)]
        if len(pred_score) == 0:
            continue
        y_true_arr.append(gt_label)
        y_pred_proba_arr.append(pred_score)
        labels_arr.append(labels_dict[i])
        plot_thres_for_idx.append(idx)
        idx = idx+1

    return y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx
    

def add_figures_to_dict(cfg, phase, figures_dict, y_true_arr, y_pred_proba_arr, labels, plot_thres_for_idx):
    if cfg["task_type"] == "multiclass-classification":
        for func in cfg["viz"]["eval"]:
            if func == "plot_pr_curve":
                fig, _ = getattr(viz_eval_module, func)(
                    y_true_arr, y_pred_proba_arr, labels, plot_thres_for_idx=plot_thres_for_idx, 
                    pos_label=1, plot_prevalance_for_idx=plot_thres_for_idx)


            else:
                fig, _ = getattr(viz_eval_module, func)(y_true_arr, y_pred_proba_arr, labels, 
                    plot_thres_for_idx=plot_thres_for_idx, pos_label=1)
            figures_dict[f"{phase}_{func}"] = fig

            

    else:
        for func in cfg["viz"]["eval"]:
            if func == "plot_pr_curve":
                fig, _ = getattr(viz_eval_module, func)(
                    y_true_arr, y_pred_proba_arr, labels, plot_thres_for_idx=plot_thres_for_idx, 
                    pos_label=1, plot_prevalance_for_idx=plot_thres_for_idx)

            else:
                fig, _ = getattr(viz_eval_module, func)(y_true_arr, y_pred_proba_arr, labels, 
                    plot_thres_for_idx=plot_thres_for_idx, pos_label=1)
            figures_dict[f"{phase}_{func}"] = fig

    return figures_dict


def convert_to_wandb_images(figures_dict):
    new_fig_dict = dict()
    for fig in figures_dict:
        new_fig_dict[fig] = [wandb.Image(figures_dict[fig])]
    # return new_fig_dict

    return figures_dict


def save_model_checkpoints(cfg, phase, metrics, ckpt_metric, ckpt_dir, model, optimizer, epochID):
    state_dict = {
        "epoch": epochID,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": ckpt_metric,
    }

    saved_best = False
    metric_key = cfg["eval"]["ckpt_metric"]
    if metrics[metric_key] > ckpt_metric:
        ckpt_metric = metrics[metric_key]
        best_model_path = os.path.join(ckpt_dir, "best_model.pth.tar")
        torch.save(state_dict, best_model_path)
        saved_best = True

    model_name = "checkpoint-{}.pth.tar".format(epochID)
    model_path = os.path.join(ckpt_dir, model_name)
    torch.save(state_dict, model_path)

    return saved_best


def init_preds_dict():
    return {
        "image_paths": [],
        "gt": [],
        "pred_label": [],
        "pred_prob": [],
    }


def parse_image_path(path):
    # Round2-TODO: Check if the path is consistent with ms1-data

    path_items = path.split("/")
    dataset = path_items[-7]
    data = path_items[-6]
    class_label = path_items[-5]
    patient = path_items[-4]
    video = path_items[-3]
    annotator_task = path_items[-2]
    image_num = path_items[-1]

    return dataset, data, class_label, patient, video, annotator_task, image_num


def cache_predictions(gt, pred_prob, pred_labels, image_paths, eval_mode, CKPT_ROOT, 
                      figures_dict=None, metrics=None):
    """

    This saves prediction outputs in a particular structure at /path/to/checkpoint/root/cache.
    These caches can be used in notebook for result analysis

    Output: The following files are cached.
    - `pred_dict` : same info structured with patientID/videoID
        ```
        pred_dict[patient][video] = {
                'image_paths' : [],   # list of all images corresponding to the patient and video ID
                'gt' : [],            # list of gt corresponding to the patient and video ID
                'pred_label' : [],    # list of pred label corresponding to the patient and video ID
                'pred_prob' : [],     # list of pred prob corresponding to the patient and video ID
            }
        ```
    """
    cache_dir = os.path.join(CKPT_ROOT, "cache_copy")  # Results cached in this dir
    os.makedirs(cache_dir, exist_ok=True)

    print("> Collating all predictions...")

    pred_dict = defaultdict(lambda: defaultdict(init_preds_dict))

    for i, image_path in enumerate(image_paths):
        _, _, _, patient, video, _, _ = parse_image_path(image_path)

        # ROUND2-TODO: Extend to frame-level and video-level probabilities.
        pred_dict[patient][video]["image_paths"].append(image_path)
        pred_dict[patient][video]["gt"].append(gt[i].item())
        pred_dict[patient][video]["pred_label"].append(pred_labels[i].item())
        pred_dict[patient][video]["pred_prob"].append(
            pred_prob[i][pred_labels[i]].item()
        )

    save_obj = [pred_dict, figures_dict, metrics]

    print("> Saving all files...")

    clf_dict_file = os.path.join(cache_dir, f"{eval_mode}_pred_dict.pkl")
    with open(clf_dict_file, "wb") as f:
        pickle.dump(save_obj, f)


def cache_predictions(gt, pred_prob, pred_labels, image_paths, eval_mode, CKPT_ROOT, 
                           figures_dict=None, metrics=None):
    cache_dir = os.path.join(CKPT_ROOT, "cache_copy")  # Results cached in this dir
    os.makedirs(cache_dir, exist_ok=True)

    print("> Collating all predictions...")

    pred_dict = dict()

    for i, image_path in enumerate(image_paths):
        image_pred_dict = dict()
        image_pred_dict["gt"] = gt[i]
        image_pred_dict["pred_label"] = pred_labels[i]
        image_pred_dict["pred_prob"] = pred_prob[i]
        image_name = image_path.split("/")[-1]
        pred_dict[image_name] = image_pred_dict

    save_obj = [pred_dict, figures_dict, metrics]

    print("> Saving all files...")

    clf_dict_file = os.path.join(cache_dir, f"{eval_mode}_pred_dict.pkl")
    with open(clf_dict_file, "wb") as f:
        pickle.dump(save_obj, f)

def classify(predict, thresholds):
    predict = max(predict, thresholds[0])
    for i in reversed(range(len(thresholds))):
        if predict >= thresholds[i]:
            return i



