from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

def train_and_validate(model, device, train_loader, val_loader, loss_fn, metric, num_epochs, optimizer, scheduler=None, verbose=True):
    train_losses, train_metric_scores = [], []
    val_losses, val_metric_scores = [], []
    
    for epoch in range(1, num_epochs + 1):
        
        model.train()
        batch_losses, batch_targets, batch_preds = [], [], []
        data_loader = tqdm(train_loader, position=0, leave=True, desc=f'Training {epoch}/{num_epochs}') if verbose else train_loader
        
        for batch in data_loader:
            encoded_atoms, edges, natoms, other_features, dGsolv = batch
            encoded_atoms, edges, other_features, dGsolv = encoded_atoms.to(device), edges.to(device), other_features.to(device), dGsolv.to(device)
            
            optimizer.zero_grad()
            dGsolv_pred = model(encoded_atoms, edges, natoms, other_features)
            loss = loss_fn(dGsolv_pred, dGsolv)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            batch_targets.extend(dGsolv.cpu().tolist())
            batch_preds.extend(dGsolv_pred.cpu().detach().tolist())
            
            if verbose:
                postfix = f'average loss = {np.array(batch_losses).mean():.3f}'
                data_loader.set_postfix_str(postfix)
        
        train_losses.append(np.array(batch_losses).mean())
        train_metric_scores.append(metric(batch_targets, batch_preds))
        
        
        model.eval()
        batch_losses, batch_targets, batch_preds = [], [], []
        data_loader = tqdm(val_loader, position=0, leave=True, desc=f'Validating {epoch}/{num_epochs}') if verbose else val_loader
        
        with torch.no_grad():
            for batch in data_loader:
                encoded_atoms, edges, natoms, other_features, dGsolv = batch
                encoded_atoms, edges, other_features, dGsolv = encoded_atoms.to(device), edges.to(device), other_features.to(device), dGsolv.to(device)

                dGsolv_pred = model(encoded_atoms, edges, natoms, other_features)
                loss = loss_fn(dGsolv_pred, dGsolv)

                batch_losses.append(loss.item())
                batch_targets.extend(dGsolv.cpu().tolist())
                batch_preds.extend(dGsolv_pred.cpu().detach().tolist())
                
                if verbose:
                    postfix = f'average loss = {np.array(batch_losses).mean():.3f}'
                    data_loader.set_postfix_str(postfix)
        
        val_losses.append(np.array(batch_losses).mean())
        val_metric_scores.append(metric(batch_targets, batch_preds))
        
        if scheduler:
            scheduler.step(val_losses[-1])
            
    return train_losses, train_metric_scores, val_losses, val_metric_scores


def plot_history(train_losses, train_metric_scores, val_losses, val_metric_scores, start_ix=0, metric_name="R2 score"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    train_losses, train_metric_scores = train_losses[start_ix:], train_metric_scores[start_ix:]
    val_losses, val_metric_scores = val_losses[start_ix:], val_metric_scores[start_ix:]

    axes[0].plot(range(1 + start_ix, len(train_losses) + start_ix + 1), train_losses, label='train loss')
    axes[0].plot(range(1 + start_ix, len(val_losses) + start_ix + 1), val_losses, label='val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(range(1 + start_ix, len(train_metric_scores) + start_ix + 1), train_metric_scores, label='train scores')
    axes[1].plot(range(1 + start_ix, len(val_metric_scores) + start_ix + 1), val_metric_scores, label='val scores')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_name)
    axes[1].legend()

    plt.show()