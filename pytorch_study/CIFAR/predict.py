import argparse
import os
from PIL import Image

import torch
import torchvision.transforms as transforms

from model import build_model
from checkpoint import load_checkpoint


CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def build_predict_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])


def predict_one_image(model, image_path, device, transform):
    image = Image.open(image_path).convert('RGB')
    x = transform(image)          # [3, 32, 32]
    x = x.unsqueeze(0)            # [1, 3, 32, 32]
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)        # [1, 10]
        pred = outputs.argmax(dim=1)
        pred_idx = pred.item()

    return pred_idx, CLASSES[pred_idx]


def main():
    parser = argparse.ArgumentParser(description='CIFAR10 single image prediction')
    parser.add_argument('--imag' \
    'e', type=str, required=True, help='path to input image')
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='path to checkpoint')
    args = parser.parse_args()

    assert os.path.exists(args.image), f'image not found: {args.image}'
    assert os.path.exists(args.checkpoint), f'checkpoint not found: {args.checkpoint}'

    device = get_device()
    print('Using device:', device)

    model = build_model(args.model).to(device)

    model, _, _, _, _ = load_checkpoint(
        args.checkpoint,
        model,
        device,
        expected_model_name=args.model
    )

    transform = build_predict_transform()

    pred_idx, pred_name = predict_one_image(model, args.image, device, transform)

    print(f'Predicted class index: {pred_idx}')
    print(f'Predicted class name: {pred_name}')


if __name__ == '__main__':
    main()