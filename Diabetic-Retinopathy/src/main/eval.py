# This script compute metrics and cache predictions on val/test set
# Flow: parse args -> get predictions -> compute metrics -> cache results
# Usage:
# PYTHONPATH='.../Diabetic-Retinopathy' \
# CUDA_VISIBLE_DEVICES=1 \
# taskset --cpu-list 10-19 \
# python src/main/eval.py --config multiclass_model.yml


import argparse
import os
import sys
import time
import torch
# import mlfoundry
sys.path.append('../..')
# print(sys.path)
# from src import utils
import src.utils.constants as constants

# pickle module cannot serialize lambda functions
from helper import (
    cache_predictions,
    create_figures,
    epoch,
    get_dataloader,
    init_wandb,
    initialise_objs,
    load_checkpoints,
    log_to_wandb,
    read_config,
    setup_checkpoint_dir,
    setup_misc_params,
)


def main(args):
    # Setting up necessary parameters
    print("Setting up parameters...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # For Automated Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # Load config
    cfg_filename = os.path.join(constants.CONFIG_DIR, args.config)
    cfg = read_config(cfg_filename)

    # Setup parameters
    setup_misc_params(cfg)

    print("Creating the dataloaders...")
    test_dataloader = get_dataloader(cfg, phase=args.eval_mode)

    print("Initializing the model...")
    # Initializing model and criterion

    model, criterion, _, _ = initialise_objs(cfg, phase="inference")
    model.to(device)
    criterion.to(device)
    
    ckpt_dir, root_dir = setup_checkpoint_dir(cfg, args, phase="inference")

    print(f"Loading from checkpoints at {ckpt_dir}")

    inference_checkpoint = cfg["inference"]["inference_checkpoint"]
    model, _, _, _ = load_checkpoints(
        cfg, args, model, None, checkpoint_id=inference_checkpoint
    )

    # Initialise W&B
    if args.wandb:
        init_wandb(cfg, args, root_dir, model, "inference")

    image_path_list = []

    print("-" * 50)
    print("Evaluating...")

    # client = mlfoundry.get_client()
    # run = client.create_run(ml_repo="ophtha-deployment")
    # model_version = run.log_model(
    #                     name="multiclass-efficientnetv2", 
    #                     model=model, 
    #                     framework="pytorch", 
    #                     step=25,  # step number, useful when using iterative algorithms like SGD
    #                 )
                    
    # print(model_version.fqn)
    # run.end()

    start_time = time.time()
    (test_loss, test_metrics, test_gt_dict, test_pred_dict) = epoch(
        cfg, model, test_dataloader, criterion, None, device, phase="inference", scaler=scaler,
        return_outputs=True)
    test_end_time = time.time()
    if "classification" in cfg['task_type']:
        test_gt_labels = test_gt_dict['gt_labels']
        test_imagepaths = test_gt_dict['imagepaths']
        test_pred_scores = test_pred_dict['pred_scores']
        test_pred_labels = test_pred_dict['pred_labels']
    elif cfg['task_type'] == "semantic-segmentation":
        test_imagepaths = test_gt_dict['imagepaths']
        test_pred_masks = test_pred_dict['pred_masks']
        test_gt_masks = test_gt_dict['gt_masks']

    print("> Time to run full epoch : {:.4f} sec".format(test_end_time - start_time))

    if "classification" in cfg['task_type']:
        labels_dict = test_dataloader.dataset.datasets[0].class_names
        labels_dict = {v: k for k, v in labels_dict.items()}
        figures_dict = create_figures(cfg, "inference", test_gt_labels=test_gt_labels, 
                                    test_scores=test_pred_scores)

    start_time = time.time()
    if args.wandb:
        log_to_wandb(figures_dict, "inference", test_metrics=test_metrics, 
                     test_loss=test_loss, epochID=0)
                     
        # wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
        #                 y_true=ground_truth, preds=predictions,
        #                 class_names=class_names)})

    print("> Time to log to W&B : {:.4f} sec".format(time.time() - start_time))

    print("Evaluation Completed!")

    cache_predictions(test_gt_labels, test_pred_scores, test_pred_labels, test_imagepaths, args.eval_mode, ckpt_dir, 
                           figures_dict=figures_dict, metrics=test_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Medical imaging model evaluation")
    parser.add_argument(
        "-c", "--config", type=str, help="Config version to be used for evaluation.",
    )
    parser.add_argument("--wandb", action="store_true", help="Use Wandb")
    parser.add_argument(
        "--eval_mode", type=str, default="test", help="Eval mode val/test"
    )
    args = parser.parse_args()
    args.config_name = os.path.splitext(args.config)[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
