from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

# Constants
DEFAULT_LR_DECAY_FACTOR = 0.1
DEFAULT_LR_THRESHOLD = 1.0


def train(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    original_model = deepcopy(model)
    current_lr = lr
    min_lr = 1e-10  # Prevent infinite loop
    
    while current_lr >= min_lr:
        model = deepcopy(original_model)
        optimizer = optim.SGD(model.parameters(), lr=current_lr, momentum=momentum)
        losses = []
        test_preds = []
        x, y = train_data
        val_x, val_y = val_data
        loss = float('Inf')
        epoch = 0
        
        while (loss > threshold) and (epoch <= epochs):
            epoch += 1
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            loss = loss.item()
            
            if (epoch % test_every_n_epochs == 0):
                with torch.no_grad():
                    new_df = pd.DataFrame({
                        'loss': F.mse_loss(model(val_x), val_y).numpy()
                    })
                    new_df['epoch'] = epoch
                    test_preds.append(new_df)
            
            # Check if we need to reduce learning rate
            if lr_tuning and ((loss > DEFAULT_LR_THRESHOLD) or np.isnan(loss)):
                current_lr = current_lr * DEFAULT_LR_DECAY_FACTOR
                print(f'Decreasing learning rate to {current_lr}')
                break  # Exit inner loop to restart with new LR
        
        # If we completed training without needing LR reduction, exit
        if not (lr_tuning and ((loss > DEFAULT_LR_THRESHOLD) or np.isnan(loss))):
            break
    
    # Final validation measurement
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': F.mse_loss(model(val_x), val_y).numpy()
        })
        new_df['epoch'] = epoch
        test_preds.append(new_df)
    
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    test_preds['split'] = 'val'
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True)