from cProfile import label
import os
import shutil
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from .val import evaluate


def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          save_dir='output',
          visual_data='Train_loss',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5):
    """
    Launch training.
    Args:
        model（nn.Layer): A sementic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        visual_data (str, optional): If use_vdl is true, then the data will be saved in this dir.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
    """
    SEED = 100
    torch.manual_seed(SEED)  # Sets the seed for generating random numbers
    # 参考https://blog.csdn.net/qq_40612314/article/details/114385936?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_antiscan_v2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_antiscan_v2&utm_relevant_index=2
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() > 0 else 'cpu')
    model.to(device)

    # 创建dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 加载模型参数
    if resume_model is not None:
        state_dict_load = torch.load(resume_model)
        model.load_state_dict(state_dict_load)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    model.train()
    if use_vdl:
        if os.path.exists(visual_data):
            shutil.rmtree(visual_data)
        os.makedirs(visual_data)
        writer = SummaryWriter(visual_data)

    avg_loss = 0.0
    mdice = 0.0
    channel_dice_array = np.array([])
    iters_per_epoch = len(train_dataloader)
    best_mean_dice = -1.0
    best_model_iter = -1
    save_models = deque()

    for iter in range(iters):
        for i, (image, label) in tqdm(enumerate(train_dataloader)):

            image = image.to(device)
            label = label.to(device)
            label = label.type(torch.int32)
            if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
                image = image.permute((0, 4, 1, 2, 3))

            logits = model(image)
            loss, per_channel_dice = losses(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO: add lr_sche
            avg_loss += loss.item()
            mdice += np.mean(per_channel_dice) * 100

            if channel_dice_array.size == 0:
                channel_dice_array = per_channel_dice
            else:
                channel_dice_array += per_channel_dice

            if (iter) % log_iters == 0:
                avg_loss /= log_iters
                mdice /= log_iters
                channel_dice_array = channel_dice_array / log_iters
                print(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, DSC: {:.4f}. "
                    .format((iter) // iters_per_epoch, iter, iters, avg_loss, mdice)
                )

            if use_vdl:
                writer.add_scalar('Loss/train', avg_loss, iter)
                writer.add_scalar('mDice/train', mdice, iter)

            avg_loss = 0.0
            mdice = 0.0
            channel_dice_array = np.array([])

            # evaluate
            if (iter % save_interval == 0 or iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0

                result_dict = evaluate(model,
                                       val_dataset,
                                       losses,
                                       num_workers=num_workers,
                                       writer=writer,
                                       print_detail=True,
                                       save_dir=save_dir)

                model.train()

            # 保存模型参数
            if iter > 0 and ((iter % save_interval) == 0 or iter == iters):
                current_save_dir = os.path.join(
                    save_dir, "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                torch.save(model.state_dict(), os.path.join(
                    current_save_dir, 'model_state_dict.pkl'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if val_dataset is not None:
                    if result_dict['mdice'] > best_mean_dice:
                        best_mean_dice = result_dict['mdice']
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        torch.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pkl'))
                    print(
                        '[EVAL] The model with the best validation mDice ({:.4f}) was saved at iter {}.'
                        .format(best_mean_dice, best_model_iter))

                    if use_vdl:
                        writer.add_scalar(
                            'Evaluate/Dice', result_dict['mdice'], iter)

                # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        writer.close()
