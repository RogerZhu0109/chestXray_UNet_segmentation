import torch
from model import UNet
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

model_path = 'unetv1.pth'

model = UNet(1, 1)
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def predict_image(img_path):
    image = Image.open(img_path)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().numpy()
    return image, output


img_path = "test/CHNCXR_0207_0.png"
original, pred = predict_image(img_path)
plt.figure(figsize=(10, 5))
# original
plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray')
plt.title('original')
plt.axis('off')
# mask
plt.subplot(1, 2, 2)
plt.imshow(pred, cmap='gray')
plt.title('mask')
plt.axis('off')

plt.tight_layout()
plt.show()
print('horray!')