from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.amp as amp
from loss import dice_loss_multiclass
from utils import dice_score_multiclass
import os 
from models import save_model

def train_model(model, train_dataloader, val_dataloader, config, verbose=True, loss_fn=None):
    device = torch.device(config['device'])
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    lr_decay_factor = config['lr_decay_factor']
    save_dir = config['save_dir']
    model_name = config['model']

    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=3)
    scaler = amp.GradScaler()

    loss_ce = nn.CrossEntropyLoss(reduction='mean')
    
    history = {
        'train_ce_loss': [], 'train_dice_loss': [], 'train_total_loss': [],
        'val_ce_loss': [], 'val_dice_loss': [], 'val_total_loss': [], 'val_mDice': []
    }

    best_val_loss = float('inf')
    print("Starting Training...")

    for epoch in range(1, n_epochs + 1):
        # ----------------------
        # Training Phase
        # ----------------------
        model.train()
        train_running_loss_ce = 0
        train_running_loss_dice = 0
        train_batch_count = 0

        for train_batch_idx, (train_inputs, train_targets) in enumerate(train_dataloader):
            train_inputs = train_inputs.to(device, dtype=torch.float32)
            train_targets = train_targets.to(device, dtype=torch.long)
            
            optimizer.zero_grad()

            with amp.autocast(device_type=device.type):
                train_preds = model(train_inputs)
                
                # CrossEntropy Loss
                train_ce_loss = loss_ce(train_preds, train_targets)
                
                # Dice Loss
                train_dice_loss = dice_loss_multiclass(train_preds, train_targets)
                
                total_loss = train_ce_loss + train_dice_loss

            # Check for NaN
            if torch.isnan(total_loss):
                print(f"NaN detected at epoch {epoch}, batch {train_batch_idx}")
                print(f"CE Loss: {train_ce_loss}, Dice Loss: {train_dice_loss}")
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_running_loss_ce += train_ce_loss.item()
            train_running_loss_dice += train_dice_loss.item()
            train_batch_count += 1

        # Compute Epoch average Loss
        avg_train_ce_loss = train_running_loss_ce / max(train_batch_count, 1)
        avg_train_dice_loss = train_running_loss_dice / max(train_batch_count, 1)
        avg_train_total_loss = avg_train_ce_loss + avg_train_dice_loss

        history['train_ce_loss'].append(avg_train_ce_loss)
        history['train_dice_loss'].append(avg_train_dice_loss)
        history['train_total_loss'].append(avg_train_total_loss)

        # ----------------------
        # Validation Phase
        # ----------------------
        model.eval()
        val_running_loss_ce = 0
        val_running_loss_dice = 0
        val_running_mDice = 0
        val_batch_count = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.to(device, dtype=torch.float32)
                val_targets = val_targets.to(device, dtype=torch.long)
                
                val_preds = model(val_inputs)
                
                val_ce_loss = loss_ce(val_preds, val_targets)
                val_dice_loss = dice_loss_multiclass(val_preds, val_targets)
                
                val_running_loss_ce += val_ce_loss.item()
                val_running_loss_dice += val_dice_loss.item()
                
                metrics = dice_score_multiclass(val_preds, val_targets)
                val_running_mDice += metrics['mDice']
                val_batch_count += 1

        avg_val_ce_loss = val_running_loss_ce / max(val_batch_count, 1)
        avg_val_dice_loss = val_running_loss_dice / max(val_batch_count, 1)
        avg_val_mDice = val_running_mDice / max(val_batch_count, 1)
        avg_val_total_loss = avg_val_ce_loss + avg_val_dice_loss

        history['val_ce_loss'].append(avg_val_ce_loss)
        history['val_dice_loss'].append(avg_val_dice_loss)
        history['val_total_loss'].append(avg_val_total_loss)
        history['val_mDice'].append(avg_val_mDice)

        os.makedirs(save_dir, exist_ok=True)

        # Judge best model
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            try:
                save_model(model, best_model_path)
            except NameError:
                torch.save(model.state_dict(), best_model_path)

        scheduler.step(avg_val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print ONCE per epoch
        if verbose:
            print(f"Epoch {epoch:2d}/{n_epochs} | LR: {current_lr:.6f} | "
                  f"Train: {avg_train_total_loss:.4f} | Val: {avg_val_total_loss:.4f} | mDice: {avg_val_mDice:.4f}")

    print("Training complete.")
    return history