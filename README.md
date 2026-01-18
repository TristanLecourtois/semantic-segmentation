# Image Segmentation 

This repository provides a way to train and evaluate state-of-the-art deep learning models for multi-class semantic segmentation on the Breast Cancer Semantic Segmentation (BCSS) dataset.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TristanLecourtois/semantic-segmentation.git
   cd semantic-segmentation
   ```

2. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   ```

## Data Structure

Ensure your dataset is organized as follows in the root directory:

```plaintext
data/BCSS/
├── train/           # RGB Images (.png, .jpg)
├── train_mask/      # Segmentation Masks (.png)
├── val/             # Validation Images
└── val_mask/        # Validation Masks
```

## Model 

You can select different architectures using the `--model` flag:

- **UNet**: Standard encoder-decoder with skip connections.  
- **ResNet34UNet**: UNet with a pre-trained ResNet34 backbone.  
- **TransUNet**: Hybrid CNN-Transformer architecture for global context.  
- **DenseUNet**: UNet based on DenseNet blocks for optimized feature reuse.  

## Training

You can launch training sessions by specifying hyperparameters.  
Below are examples for different configurations.

### Option A: Standard U-Net Training

Train a classic U-Net with 64 filters and specific learning rate decay.

```bash
python main.py --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 \
    --model UNet --save_dir ./checkpoints/unet_64n --n_filters 64 --device cuda
```

### Option B: Other Architectures (ResNet / TransUNet...)

Use a pre-trained ResNet34 backbone or a Transformer-based model for higher accuracy.

```bash
# Train ResNet34-UNet
python main.py --n_epochs 30 --batch_size 16 --learning_rate 1e-4 \
    --model ResNet34UNet --save_dir ./checkpoints/resnet_exp1 --device cuda

# Train TransUNet
python main.py --n_epochs 40 --batch_size 8 --learning_rate 5e-4 \
    --model TransUNet --save_dir ./checkpoints/transunet_exp1 --img_size 224 --device cuda
```

## 🛠️ Arguments Breakdown

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | ResNet34UNet | Selection: UNet, ResNet34UNet, TransUNet, DenseUNet |
| `--n_epochs` | int | 15 | Total number of training epochs |
| `--batch_size` | int | 128 | Number of samples per batch |
| `--learning_rate` | float | 5e-3 | Initial LR for AdamW optimizer |
| `--lr_decay_factor` | float | 0.85 | Decay rate for ReduceLROnPlateau |
| `--output_channels` | int | 3 | Number of segmentation classes (BCSS default) |
| `--save_dir` | str | ./checkpoints/ | Folder to store weights and logs |
| `--data_path` | str | ./data/BCSS | Path to the dataset root |
| `--device` | str | cuda | Hardware to use (cuda or cpu) |

## Results

After training, the following files will be generated in your `--save_dir`:

- Best Model: `<model_name>_best.pth` (lowest validation loss)  
- Learning Curves: PNG files for Cross-Entropy, Dice Loss, and mDice scores  
- Visual Check: A plot showing original images, ground truth masks, and model predictions will be automatically generated.


