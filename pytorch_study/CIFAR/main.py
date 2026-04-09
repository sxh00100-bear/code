'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import os
from engine import train, evaluate, test
from data import get_data_loaders
import argparse
from checkpoint import save_checkpoint, load_checkpoint
from model import build_model

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ResNet Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-workers', type=int, default=2, 
                        help='number of dataloader workers')
    parser.add_argument('--save-dir', type=str, default='checkpoints', 
                        help='directory to save checkpoints')
    parser.add_argument('--model', type=str, default='ResNet18', metavar='M',
                        help='choose the Model')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='resume from checkpoint')
    parser.add_argument('--data-root', type=str, default='./data', 
                        help='dataset root directory')
    args = parser.parse_args()

    #device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using device:', device)

    #data
    trainloader, valloader, testloader = get_data_loaders(batch_size=args.batch_size,
                                                          num_workers=args.num_workers,
                                                          data_root=args.data_root)

    #model
    model = build_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    
    #train
    best_acc = 0.0
    start_epoch = 1
    
    save_dir = os.path.join(args.save_dir, args.model)
    best_path = os.path.join(save_dir, 'best.pth')
    last_path = os.path.join(save_dir, 'last.pth')
    if args.resume is not None:
        model, optimizer, scheduler, start_epoch, best_acc = load_checkpoint(
            args.resume, model, device, optimizer, scheduler, expected_model_name=args.model
        )
        print(f'Resumed from {args.resume}, start_epoch={start_epoch}, will train {args.epochs} more epochs, best_acc={best_acc:.2f}')

    end_epoch = start_epoch + args.epochs 
    for epoch in range(start_epoch, end_epoch):
        train_loss,train_acc=train(model, trainloader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, valloader, device, criterion)
        
        print('Epoch: %d | train Loss: %.3f | train Acc: %.3f%%' % 
                  (epoch, train_loss, train_acc))
        print ('Val Loss: %.3f | Val Acc: %.3f%%' % (val_loss, val_acc))
        
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'model_name': args.model,
            }, best_path)
        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'model_name': args.model,
        }, last_path)

        scheduler.step() 


    #test with best checkpoint
    
    assert os.path.exists(best_path), 'best checkpoint not found'                      
    model, _, _, _, _ = load_checkpoint(best_path, model, device,expected_model_name=args.model)    
    avg_test_loss, test_acc = test(model, testloader, device, criterion)
    print(f'Final Test Loss: {avg_test_loss:.4f} | Final Test Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    main()