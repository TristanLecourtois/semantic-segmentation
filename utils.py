import matplotlib.pyplot as plt
import numpy as np
import torch
import os 
import torch.nn.functional as F

def visualize_sample(dataloader, n_samples=1,save_dir="plots"):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images, masks = next(iter(dataloader))
    
    for i in range(n_samples):

        # denormalize image using mean and std from transformations 
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().numpy()

        plt.figure(figsize=(15, 5))

        # original image 
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"Sample {i+1}: Original RGB")
        plt.axis("off")

         # segmentation mask 
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="tab10", interpolation="nearest")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        #overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.imshow(mask, cmap="tab10", alpha=0.4, interpolation="nearest")
        plt.title("Overlay Visualization")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"sample_visualization_{i+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


def dice_score_multiclass(y_pred, y_true, smooth=1e-6, bg=False, ignore_index=255):
    """
    Robust Dice Score for High Cardinality Classes
    """
    with torch.no_grad(): 
        y_pred_idx = torch.argmax(y_pred, dim=1)
        

        valid_mask = (y_true != ignore_index)
        y_pred_idx = y_pred_idx[valid_mask]
        y_true = y_true[valid_mask]
        
        C = y_pred.shape[1]
        results = {}
        foreground_scores = [] 

        for i in range(C):
            p_mask = (y_pred_idx == i)
            t_mask = (y_true == i)

            intersection = (p_mask & t_mask).sum().float()
            union = p_mask.sum() + t_mask.sum()


            if union == 0:
                score = 1.0 # pred and ground_truth are 0，perfect !
            else:
                dice = (2 * intersection + smooth) / (union + smooth)
                score = dice.item() 

            results[f"Dice_Class_{i}"] = score

            # mDice 
            if i > 0:
                foreground_scores.append(score)
            elif i == 0 and bg:
                foreground_scores.append(score)

        if len(foreground_scores) > 0:
            results["mDice"] = sum(foreground_scores) / len(foreground_scores)
        else:
            results["mDice"] = 0.0
    return results



####### mIou #######
def iou_multiclass(y_pred, y_true,smooth=1e-6,bg =False):

    y_pred_idx = torch.argmax(y_pred, dim=1)

    C = y_pred.shape[1]
    y_pred_1hot = F.one_hot(y_pred_idx, C).permute(0, 3, 1, 2).float()
    y_true_1hot = F.one_hot(y_true, C).permute(0, 3, 1, 2).float()

    class_ious = {}
    foreground_ious = []
    for i in range(C):
        # Flatten Global Batch IoU
        p = y_pred_1hot[:, i].contiguous().view(-1)
        t = y_true_1hot[:, i].contiguous().view(-1)

        intersection = (p * t).sum()
        # IoU 分母 = A + B - (A ∩ B)
        union = p.sum() + t.sum() - intersection

        if union == 0:
            iou = 1.0 # Empty target & Empty pred = Perfect
        else:
            iou_tensor = (intersection + smooth) / (union + smooth)
            iou = iou_tensor.item()

        class_ious[f"class_{i}_iou"] = iou

    # 4. Compute mIoU (Mean IoU)
    if not bg:
        foreground_ious = [v for k, v in class_ious.items() if "class_0" not in k]
    elif bg:
        foreground_ious = [v for k, v in class_ious.items()]

    if len(foreground_ious) > 0:
        m_iou = sum(foreground_ious) / len(foreground_ious)
    else:
        m_iou = 0.0
        
    class_ious["mIoU"] = m_iou
    
    return class_ious

def plot_learning_curves(train_loss, val_loss, title='--------', ylabel='Loss', save_path='./ckpt'):
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = '#333333'

    fig, axis = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training and validation loss (NaN is used to offset epochs by 1)
    if train_loss is not None and len(train_loss) > 0:
        axis.plot([np.NaN] + train_loss, 
                  marker='o', linestyle='-', linewidth=2, 
                  markersize=5, label='Training Loss')
        
    axis.plot([np.NaN] + val_loss, marker='s', linestyle='-', linewidth=2, markersize=5, label=f'Validation {ylabel}')

    # Adding title, labels and formatting
    axis.set_title(title, fontsize=16)
    axis.set_xlabel('Epoch', fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14, rotation=0, labelpad=20)
    
    axis.legend(fontsize=12)
    axis.grid(True, which='both', linestyle='--', linewidth=0.5)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{title}.png')
    plt.show()


def visualize_prediction(model, dataloader, device, num_samples=3):
    model.eval()
    inputs, targets = next(iter(dataloader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1) # [B, H, W]
        
    # Turn to CPU
    inputs = inputs.cpu()
    targets = targets.cpu()
    preds = preds.cpu()
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        # Original figure
        # Suppose inputs is [C, H, W]
        img = inputs[i].permute(1, 2, 0).numpy()
        # Process range
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(targets[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(preds[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 2].set_title("Model Prediction")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

def display_test_sample(model, test_input, test_target, device):
    model.eval()
    test_input, test_target = test_input.to(device), test_target.to(device)

    with torch.no_grad():
        logits = model(test_input)
        probs = torch.softmax(logits, dim=1)                # [B, C, H, W]
        pred_mask = torch.argmax(probs, dim=1)              # [B, H, W]

    # convert to numpy
    image       = test_input[0].detach().cpu().permute(1,2,0).numpy()
    gt_mask     = test_target[0].detach().cpu().numpy()
    pred_mask   = pred_mask[0].detach().cpu().numpy()

    plt.rcParams['figure.facecolor'] = '#171717'
    plt.rcParams['text.color']       = '#DDDDDD'
    plt.figure(figsize=(15,5))

    # image
    plt.subplot(1,3,1)
    plt.title("H&E Image")
    plt.imshow(image.astype(np.uint8))
    plt.axis("off")

    # GT mask
    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap="tab20")
    plt.axis("off")

    # pred mask
    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="tab20")
    plt.axis("off")

    plt.show()

    # overlay 
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.title("Overlay: Ground Truth")
    plt.imshow(image.astype(np.uint8))
    plt.imshow(gt_mask, cmap="tab20", alpha=0.35)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Overlay: Prediction")
    plt.imshow(image.astype(np.uint8))
    plt.imshow(pred_mask, cmap="tab20", alpha=0.35)
    plt.axis("off")

    plt.show()


def visualize_two_models(
    model1, model2,
    dataloader, device,
    num_samples=5, alpha=0.4,
    save_path=None,
    model1_name="Model A",
    model2_name="Model B"
):
    model1.eval()
    model2.eval()
    
    images, masks = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        logits1 = model1(images)
        preds1 = torch.argmax(torch.softmax(logits1, dim=1), dim=1)
        
        logits2 = model2(images)
        preds2 = torch.argmax(torch.softmax(logits2, dim=1), dim=1)
    
    images = images.cpu()
    masks = masks.cpu()
    preds1 = preds1.cpu()
    preds2 = preds2.cpu()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 5 * num_samples))
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        gt = masks[i].numpy()
        p1 = preds1[i].numpy()
        p2 = remap_classes(preds2[i].numpy())

        # Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # GT
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(gt, cmap='jet', alpha=alpha)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Model 1
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(p1, cmap='jet', alpha=alpha)
        axes[i, 2].set_title(model1_name)
        axes[i, 2].axis('off')
        
        # Model 2
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(p2, cmap='jet', alpha=alpha)
        axes[i, 3].set_title(model2_name)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    
    plt.show()

