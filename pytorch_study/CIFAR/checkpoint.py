import torch
import os

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, device, optimizer=None, scheduler=None,expected_model_name=None):
    checkpoint = torch.load(path, map_location=device)
    if expected_model_name is not None:
        ckpt_model_name = checkpoint.get('model_name', None)
        if ckpt_model_name is not None and ckpt_model_name != expected_model_name:
            raise ValueError(
                f'Checkpoint model_name={ckpt_model_name}, but expected {expected_model_name}'
            )
    model.load_state_dict(checkpoint['model'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    return model, optimizer, scheduler, start_epoch, best_acc