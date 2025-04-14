import torch
import torch.nn as nn
from tqdm import tqdm


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)  # flatten the tensors into 1D vectors
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)  # 2*intersection/total
        return 1 - dice


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_loss, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc='training loop'):
            optimizer.zero_grad()  # clears previous gradients
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()  # backpropagation
            optimizer.step()  # update model's weights
            running_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='validation loop'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_loss.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print('epoch: ', epoch, ' TrainLoss: ', avg_train_loss, ' Val Loss', avg_val_loss)

    return train_loss, val_losses
