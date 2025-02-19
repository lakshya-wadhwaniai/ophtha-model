import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import tree
from PIL import Image
# import gpytorch

# from coral_pytorch.dataset import (levels_from_labelbatch, proba_to_label, corn_label_from_logits)

from .postprocess import calculate_metrics, classify


def epoch(cfg, model, dataloader, criterion, optimizer, device, phase, scaler, return_outputs=False, use_threshold_model = False, clf=False):
    """
    This function implements one epoch of training or evaluation

    Args:
        cfg (dict): Configuration file used to run the code.
        model (pytorch model): Network architecture which is being trained.
        dataloader (pytorch Dataloader): Train/ Val/ Test dataloader.
        criterion (pytorch Criterion): Loss function specified in the config file.
        optimizer (pytorch Optimizer): Optimizer specified in the config file.
        device (pytorch device): Specifies whether to run on CPU or GPU.
        phase (str): Run a training loop or inference loop. Should be one of "train" or "inference".
        scaler (scaler object): Scaler used during AMP training.

    Returns:
        (float, dict, list, list, list, list): A tuple containing the average loss, metrics computed, ground truth labels, predicted score, predicted labels, and a list containing the file paths of all data samples encountered in the epoch.
    """

    print("*" * 15 + f" || {phase} || " + "*" * 15)
    if phase == "train":
        model.train()
    else:
        model.eval()

    losses = []
    batch_times = []
    imagepaths_list = []

    gt_labels_list = []
    pred_scores_list = []
    feat_list = []
    symbol = "#"
    width = 40
    # features = False
    total = len(dataloader)

    tick = time.time() 

#abstraction of loss calculation
    for batchID, batch_tuple in enumerate(dataloader):
        
        images = batch_tuple[0]
        labels = batch_tuple[1]
        imagepaths = batch_tuple[2]
        tock = time.time()
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=cfg[phase]["use_amp"]):
            with torch.set_grad_enabled(phase == "train"):
                output = model(images)
                if (
                    cfg["model"]["name"] == "inception"
                    or cfg["model"]["name"] == "googlenet"
                    or cfg["model"]["name"] == "resnet50_regression_features"
                ):
                    features = output[1]
                    output = output[0]

                if cfg['task_type'] == 'multilabel-classification':
                    loss = criterion(output, labels.to(torch.float))
                    uncertain_labels = cfg['data']['ignore_labels']
                    uncertain_labels = np.array(uncertain_labels)
                    nan_labels = cfg['data']['nan_labels']
                    nan_labels = np.array(nan_labels)
                    ignore_idx = np.where(np.isin(labels.detach().cpu(),[uncertain_labels,nan_labels]))
                    loss[ignore_idx] = np.nan
                    loss = torch.nanmean(loss)

                elif cfg['task_type'] == 'multiclass-classification':
                    if ((cfg['train']['loss'] == 'MSELoss') or (cfg['train']['loss'] == 'SmoothL1Loss')) and ("regression" in cfg['model']['name']):
                        loss = criterion(output, (labels.to(torch.float)).reshape(-1,1))
                    elif cfg['train']['loss'] == 'MSELoss':
                        softmax = nn.Softmax(dim=1)
                        loss = criterion(softmax(output), (F.one_hot(labels, num_classes=5)).to(torch.float))
                    elif cfg['train']['loss'] == 'CoralLoss':
                        loss = criterion(torch.sigmoid(output), levels_from_labelbatch(labels, num_classes= 5).to(device))
                    
                    elif cfg['train']['loss'] == 'CornLoss':
                        loss = criterion(output, labels)
                    else:
                        loss = criterion(output, labels)
                    loss = torch.nanmean(loss)
                
                elif cfg['task_type'] == 'binary-classification':
                    output = output.reshape((-1,))
                    loss = criterion(output, labels.type(torch.float32).to(device))
                    loss = torch.nanmean(loss)

        current = batchID + 1
        percent = current / float(total)
        size = int(width * percent)
        batch_time = tock - tick
        bar = "[" + symbol * size + "." * (width - size) + "]"

        if cfg['task_type'] == 'multiclass-classification':

            accuracy = (torch.sum(labels == torch.argmax(output, axis = 1)))/output.shape[0]

            print(
                "\r Data={:.4f} s | ({:4d}/{:4d}) | {:.2f}% | Loss={:.4f} | Accuracy={:.4f} {}".format(
                    batch_time, current, total, percent * 100, loss.item(), accuracy.item(), bar
                ),
                end="",
            )
        
        elif cfg['task_type'] == 'binary-classification':


            print(
                "\r Data={:.4f} s | ({:4d}/{:4d}) | {:.2f}% | Loss={:.4f} | {}".format(
                    batch_time, current, total, percent * 100, loss.item(), bar
                ),
                end="",
            )

        losses.append(loss.item())
        batch_times.append(batch_time)

        if phase == "train":
            # Setting grad=0 using another method instead of optimizer.zero_grad()
            # See: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            for param in model.parameters():
                param.grad = None
            if cfg[phase]["use_amp"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        # Append GT Masks and Pred Masks
        labels = labels.detach().cpu()
        output = output.detach().cpu()

        if "features" in cfg["model"]["name"]:
            features = features.detach().cpu()
            feat_list += features


        gt_labels_list += labels
        pred_scores_list += output
        imagepaths_list.extend(imagepaths)
        tick = time.time()

    # Metric Calculation
    gt_labels = torch.stack(gt_labels_list, dim=0).numpy()
    pred_scores = torch.stack(pred_scores_list, dim=0).to(torch.float32)

    if "features" in cfg["model"]["name"]:
        features = torch.stack(feat_list, dim=0).to(torch.float32)


    if (cfg["task_type"] == "binary-classification"):
        pred_scores = torch.sigmoid(pred_scores).numpy()
        pred_labels = torch.zeros(gt_labels.shape).numpy()

        pred_labels[(pred_scores > cfg['inference']['threshold'])] = 1.0

        pred_labels = pred_labels.astype(np.int8)
        gt_labels = gt_labels.astype(np.int8)

    elif cfg["task_type"] == "multiclass-classification" and ("regression" in cfg["model"]["name"]) and (use_threshold_model == False):
        threshold = cfg['inference']['threshold']
        pred_labels = torch.tensor(
                [classify(p, threshold) for p in pred_scores]
            ).float()

    elif cfg["task_type"] == "multiclass-classification" and ("regression" in cfg["model"]["name"]):
        threshold = cfg['inference']['threshold']
        if 'features' in cfg['model']['name']:
            print('using features')
            pred_labels = torch.tensor(use_threshold_model.predict(features))
        else:
            print('using scores')
            pred_labels = torch.tensor(use_threshold_model.predict(pred_scores))

    elif cfg["task_type"] == "multiclass-classification" and cfg["train"]["loss"] == 'CoralLoss':
        pred_scores = torch.sigmoid(pred_scores)
        pred_labels = proba_to_label(pred_scores).float()

    elif cfg["task_type"] == "multiclass-classification" and cfg["train"]["loss"] == 'CornLoss':
        pred_labels = corn_label_from_logits(pred_scores).float()

    elif cfg["task_type"] == "multiclass-classification":
        pred_labels = torch.argmax(pred_scores, axis=1).numpy()

    elif cfg["task_type"] == "multilabel-classification":
        pred_scores = torch.sigmoid(pred_scores).numpy()
        pred_labels = np.where(np.isnan(pred_scores), np.nan,
                               (pred_scores > 0.5).astype(np.int8))

    pred_dict = {
        'pred_scores': pred_scores,
        'pred_labels': pred_labels
    }

    gt_dict = {
        'gt_labels': gt_labels, 
        'imagepaths': imagepaths_list
    }
    
    metrics_dict = calculate_metrics(cfg, gt_dict, pred_dict, phase, dataloader)

    average_loss = np.mean(losses)
    average_time = np.mean(batch_times)

    bar = "[" + symbol * width + "]"
    print(
        "\rData={:.4f} s | ({:4d}/{:4d}) | 100.00% | Loss={:.4f} {}".format(
            average_time, total, total, average_loss, bar
        )
    )

    return average_loss, metrics_dict, gt_dict, pred_dict
