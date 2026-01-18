import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import BCSSDataset, train_transform, val_transform
from models import UNet, ResNet34UNet, TransUNet, DenseUNet, count_parameters, save_model
from train import train_model
from utils import plot_learning_curves, visualize_prediction

def get_args():
    parser = argparse.ArgumentParser(description="Train Segmentation Models on BCSS Dataset")
    
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--lr_decay_factor', type=float, default=0.85, help='Facteur de réduction du scheduler')
    parser.add_argument('--model', type=str, default='UNet', 
                        choices=['UNet', 'ResNet34UNet', 'TransUNet', 'DenseUNet'])
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--output_channels', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=224)
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./data/BCSS')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device (cuda ou cpu)')

    return parser.parse_args()

def main():

    args = get_args()
    config = vars(args)

    TRAIN_IMG_DIR = os.path.join(args.data_path, "train")
    TRAIN_MASK_DIR = os.path.join(args.data_path, "train_mask")
    VAL_IMG_DIR = os.path.join(args.data_path, "val")
    VAL_MASK_DIR = os.path.join(args.data_path, "val_mask")

    train_ds = BCSSDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    val_ds = BCSSDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print(f"Device: {args.device} | Epochs: {args.n_epochs} | Batch Size: {args.batch_size}")

    if args.model == 'UNet':
        model = UNet(in_channels=3, out_channels=args.output_channels, n_filters=args.n_filters)
    elif args.model == 'ResNet34UNet':
        model = ResNet34UNet(in_channels=3, out_channels=args.output_channels, pretrained=True)
    elif args.model == 'TransUNet':
        model = TransUNet(num_channels=3, num_classes=args.output_channels, image_size=args.img_size)
    elif args.model == 'DenseUNet':
        model = DenseUNet(num_channels=3, num_classes=args.output_channels)

    model.to(args.device)
    count_parameters(model)

    history = train_model(model, train_loader, val_loader, config, verbose=True)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path_last = os.path.join(args.save_dir, f"{args.model}_last.pth")
    torch.save(model.state_dict(), save_path_last)

    plot_learning_curves(history['train_total_loss'], history['val_total_loss'], 
                         title=f'{args.model} Total Loss', save_path=args.save_dir)
    
    plot_learning_curves(None, history['val_mDice'], 
                         title=f'{args.model} mDice Score', save_path=args.save_dir, ylabel='Dice')

    visualize_prediction(model, val_loader, args.device, num_samples=4)

if __name__ == "__main__":
    main()