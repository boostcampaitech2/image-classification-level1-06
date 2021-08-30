import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TestDataset


def load_model(model_name, pth_name, saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        model_arch=model_name,
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, pth_name)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model = load_model(args.model_name, args.pth_name, model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'faces')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    test_dataset = TestDataset(img_paths, args.resize)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for idx, images in enumerate(pbar):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output_{args.pth_name}.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=120, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(280, 210), help='resize size for image when you trained (default: (280, 210))')
    parser.add_argument('--model', type=str, default='Model', help='model type (default: BaseModel)')
    parser.add_argument('--model_name', type=str, default='efficientnet_b4', help='what kinds of models (default: efficientnet_b4)')
    parser.add_argument('--pth_name', type=str, default='', help='which pth you will use (not optional)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', ''))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join('./model', args.model_dir)
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    assert args.pth_name, "적용하고자 하는 모델 파라미터를 입력해주세요"
    assert args.model_dir, "기본경로로 ./model 이 설정되어 있습니다. 하위 경로를 추가로 입력해주세요."
    inference(data_dir, model_dir, output_dir, args)
