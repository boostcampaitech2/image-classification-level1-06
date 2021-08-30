import argparse
import json
import multiprocessing
import os
from PIL import Image
import random
import re
import pandas as pd
from glob import glob
from importlib import import_module
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ValidDataset
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = ValidDataset.decode_multi_class(gt)
        pred_decoded_labels = ValidDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def labeling(x, df):
    path, sub_label = x['path'], x['sub_label']
    for image_path in glob(os.path.join('/opt/ml/input/data/train/faces', path, '*')):
        if 'incorrect' in image_path: new_label = 6
        elif 'normal' in image_path: new_label = 12
        else: new_label = 0
        label = x.age_label + x.gender_label + new_label
        df.append([image_path, x.age_label, x.gender_label, new_label, label])


def get_img_stats(img_paths):
    img_info = dict(means=[], stds=[])
    for img_path in img_paths:
        img = np.array(Image.open(glob(img_path)[0]).convert('RGB'))
        img_info['means'].append(img.mean(axis=(0,1)))
        img_info['stds'].append(img.std(axis=(0,1)))
    return img_info


def train(model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_classes = 18

    # -- dataset
    if args.data_changed:
        data = pd.read_csv('./input/data/train/train.csv')
        data['age_label'] = data['age'].apply(lambda x: int(int(x) >= 30) + int(int(x) >= 58))
        data['gender_label'] = data['gender'].apply(lambda x: int(len(x) * 1.5 - 6))
        data['sub_label'] = data.apply(lambda x: x.age_label + x.gender_label, axis=1)
        train_df, valid_df = train_test_split(data, test_size=args.val_ratio, shuffle=True,
                                              stratify=data['sub_label'], random_state=args.seed)
        df = []
        train_df.apply(lambda x : labeling(x, df), axis=1)
        train_df = pd.DataFrame(data=df, columns=['path', 'age_label', 'gender_label', 'mask_label', 'label'])

        df = []
        valid_df.apply(lambda x: labeling(x, df), axis=1)
        valid_df = pd.DataFrame(data=df, columns=['path', 'age_label', 'gender_label', 'mask_label', 'label'])

        train_df.to_csv(args.train_df, index=False)
        valid_df.to_csv(args.valid_df, index=False)

        img_stats = get_img_stats(train_df.path.values)
        mean = np.mean(img_stats["means"], axis=0) / 255.
        std = np.mean(img_stats["stds"], axis=0) / 255.

    train_dataset_module = getattr(import_module("dataset"), 'TrainDataset')
    train_dataset = train_dataset_module(
        train_df_path=args.train_df
    )
    if args.data_changed:
        train_dataset.mean = mean
        train_dataset.std = std

    valid_dataset_module = getattr(import_module("dataset"), 'ValidDataset')
    valid_dataset = valid_dataset_module(
        valid_df_path=args.valid_df
    )

    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = train_transform_module(
        resize=args.resize,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    train_dataset.set_transform(transform)

    base_transform_module = getattr(import_module("dataset"), 'BaseAugmentation')
    transform = base_transform_module(
        resize=args.resize,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )
    valid_dataset.set_transform(transform)


    # -- data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: Model
    model = model_module(
        model_arch=args.model_name,
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Cyclic
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        # weight_decay=5e-4,
    )
    scheduler = CyclicLR(
        optimizer,
        base_lr=args.lr,
        max_lr=1e-6,
        step_size_down=len(train_dataset) * 2 // args.batch_size,
        step_size_up=len(train_dataset) // args.batch_size,
        cycle_momentum=False,
        mode="triangular2")
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_train_acc = best_valid_acc = 0
    best_train_loss = best_valid_loss = np.inf
    best_train_f1 = best_valid_f1 = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        train_batch_loss = []
        train_batch_accuracy = []
        train_batch_f1 = []
        pbar = tqdm(train_loader)
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            if np.random.random() <= args.cutmix:
                W = inputs.shape[2]
                mix_ratio = np.random.beta(1, 1)
                cut_W = np.int(W * mix_ratio)
                bbx1 = np.random.randint(W - cut_W)
                bbx2 = bbx1 + cut_W

                rand_index = torch.randperm(len(inputs))
                target_a = labels
                target_b = labels[rand_index]

                inputs[:, :, :, bbx1:bbx2] = inputs[rand_index, :, :, bbx1:bbx2]
                outs = model(inputs)
                loss = criterion(outs, target_a) * mix_ratio + criterion(outs, target_b) * (1. - mix_ratio)
            else:
                outs = model(inputs)
                loss = criterion(outs, labels)

            preds = torch.argmax(outs, dim=-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_loss.append(
                loss.item()
            )
            train_batch_accuracy.append(
                (preds == labels).sum().item() / args.batch_size
            )
            f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
            train_batch_f1.append(
                f1
            )

            pbar.set_description(
                f'Epoch #{epoch:2f}\n'
                f'train | f1 : {train_batch_f1[-1]:.5f} | accuracy : {train_batch_accuracy[-1]:.5f} | '
                f'loss : {train_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
            )

            if (idx + 1) % args.log_interval == 0:
                train_loss = sum(train_batch_loss[idx-args.log_interval:idx]) / args.log_interval
                train_acc = sum(train_batch_accuracy[idx-args.log_interval:idx]) / args.log_interval
                train_f1 = sum(train_batch_f1[idx-args.log_interval:idx]) / args.log_interval

                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1-score", train_f1, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/learing_rate", get_lr(optimizer), epoch * len(train_loader) + idx)
            scheduler.step()

        train_item = (sum(train_batch_loss) / len(train_loader),
                      sum(train_batch_accuracy) / len(train_loader),
                      sum(train_batch_f1) / len(train_loader))
        best_train_loss = min(best_train_loss, train_item[0])
        best_train_acc = max(best_train_acc, train_item[1])
        best_train_f1 = max(best_train_f1, train_item[2])
        
        # val loop
        with torch.no_grad():
            model.eval()
            valid_batch_loss = []
            valid_batch_accuracy = []
            valid_batch_f1 = []
            figure = None
            pbar = tqdm(valid_loader, total=len(valid_loader))
            for (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                valid_batch_loss.append(
                    criterion(outs, labels).item()
                )
                valid_batch_accuracy.append(
                    (labels == preds).sum().item() / args.valid_batch_size
                )
                f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
                valid_batch_f1.append(
                    f1
                )

                pbar.set_description(
                    f'valid | f1 : {valid_batch_f1[-1]:.5f} | accuracy : {valid_batch_accuracy[-1]:.5f} | '
                    f'loss : {valid_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
                )

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = valid_dataset.denormalize_image(inputs_np, valid_dataset.mean, valid_dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=True
                    )

            valid_item = (sum(valid_batch_loss) / len(valid_loader),
                          sum(valid_batch_accuracy) / len(valid_loader),
                          sum(valid_batch_f1) / len(valid_loader))
            best_valid_loss = min(best_valid_loss, valid_item[0])
            best_valid_acc = max(best_valid_acc, valid_item[1])
            best_valid_f1 = max(best_valid_f1, valid_item[2])
            cur_f1 = valid_item[2]

            if cur_f1 >= 0.7:
                if cur_f1 > best_valid_f1:
                    print(f"New best model for valid f1 : {cur_f1:.5%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_{cur_f1:.4f}.pth")
                    best_valid_f1 = cur_f1
                else:
                    torch.save(model.module.state_dict(), f"{save_dir}/last_{cur_f1:.4f}.pth")

            print(
                f"[Train] f1 : {train_item[2]:.5}, best f1 : {best_train_f1:.5} || " 
                f"acc : {train_item[1]:.5%}, best acc: {best_train_acc:.5%} || "
                f"loss : {train_item[0]:.5}, best loss: {best_train_loss:.5} || "
            )
            print(
                f"[Valid] f1 : {valid_item[2]:.5}, best f1 : {best_valid_f1:.5} || "
                f"acc : {valid_item[1]:.5%}, best acc: {best_valid_acc:.5%} || "
                f"loss : {valid_item[0]:.5}, best loss: {best_valid_loss:.5} || "
            )

            logger.add_scalar("Val/loss", valid_item[2], epoch)
            logger.add_scalar("Val/accuracy", valid_item[1], epoch)
            logger.add_scalar("Val/f1-score", valid_item[0], epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=24, help='number of epochs to train (default: 1)')
    parser.add_argument('--augmentation', type=str, default='TrainAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[280, 210], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=30, help='input batch size for training (default: 30)')
    parser.add_argument('--valid_batch_size', type=int, default=120, help='input batch size for validing (default: 120)')
    parser.add_argument('--model', type=str, default='Model', help='model class (default: BaseModel)')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4', help='what kinds of models (default: efficientnet_b4)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--cutmix', type=float, default='0.5', help='cutmix ratio (if ratio is 0, not cutmix)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=21, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='experiment', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--train_df', type=str, default="/opt/ml/train_stratified_face.csv",
                        help='csv file path of train data')
    parser.add_argument('--valid_df', type=str, default="/opt/ml/valid_stratified_face.csv",
                        help='csv file path of validation data')
    parser.add_argument('--data_changed', type=bool, default=False,
                        help='change data and settings (default: False)')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/faces'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(model_dir, args)
