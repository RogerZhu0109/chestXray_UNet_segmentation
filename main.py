import torch
import torchvision.transforms as transforms
from torch.optim import AdamW
from data_loading import get_loaders
from model import UNet
from train import train, DiceLoss
import matplotlib.pyplot as plt

IMG_DIR = 'CXR_png'
MASK_DIR = 'masks'


def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_loader, val_loader = get_loaders(IMG_DIR, MASK_DIR, 16, transform, 0.2)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    model = UNet(1, 1)
    criterion = DiceLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)

    num_epochs = 10
    train_loss, val_loss = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    torch.save(model.state_dict(), "unetv1.pth")

    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_loss, label='training loss', color='b')
    plt.plot(epochs, val_loss, label='val loss', color='r')
    plt.show()


if __name__ == "__main__":
    main()
