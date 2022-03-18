import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from .infer import inference
import torch.nn.functional as F


def evaluate(model,
             eval_dataset,
             losses,
             num_workers=0,
             print_detail=True,
             writer=None,
             save_dir=None):
    """
    Launch evalution.
    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (torch.utils.data.Dataset): Used to read and process validation datasets.
        losses(dict): Used to calculate the loss. e.g: {"types":[loss_1...], "coef": [0.5,...]}
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric.
        writer: visualdl log writer.
        save_dir(str, optional): the path to save predicted result.
    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() > 0 else 'cpu')

    loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    total_iters = len(loader)
    logits_all = None
    label_all = None

    if print_detail:
        print(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))

    mdice = 0.0
    channel_dice_array = np.array([])
    loss_all = 0.0

    with torch.no_grad():
        for iter, (im, label, idx) in enumerate(loader):
            label = label.astype('int32')
            label = label.to(device)

            pred, logits = inference(  # reverse transform here
                model,
                im,
                ori_shape=label.shape[-3:],
                transforms=eval_dataset.transforms.transforms)

            # Post process
            # if eval_dataset.post_transform is not None:
            #     pred, label = eval_dataset.post_transform(
            #         pred.numpy(), label.numpy())
            #     pred = torch.tensor(pred)
            #     label = torch.tensor(label)

            # logits [N, num_classes, D, H, W]
            loss, per_channel_dice = losses(logits, label)
            loss = sum(loss)

            loss_all += loss.numpy()
            mdice += np.mean(per_channel_dice)
            if channel_dice_array.size == 0:
                channel_dice_array = per_channel_dice
            else:
                channel_dice_array += per_channel_dice

    mdice /= total_iters
    channel_dice_array /= total_iters
    loss_all /= total_iters

    result_dict = {"mdice": mdice}

    if print_detail:
        infor = "[EVAL] #Images: {}, Dice: {:.4f}, Loss: {:6f}".format(
            len(eval_dataset), mdice, loss_all[0])
        print(infor)
        print("[EVAL] Class dice: \n" +
              str(np.round(channel_dice_array, 4)))

    return result_dict
