import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = UNetBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))

def segment_leaf(image_path, model_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            os.getcwd(), 'classifier', 'static', 'classifier', 'seg_outputs', 'leaf_mask.png'
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Input image path: {image_path}")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet().to(device)
    print("Loading model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")

    print("Loading and preprocessing image...")
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    print(f"Image size: {image_np.shape}")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    augmented = transform(image=image_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    print("Running inference...")
    with torch.no_grad():
        pred = model(input_tensor)
        pred_binary = (pred > 0.5).float()
    print(f"Prediction shape: {pred.shape}")

    mask_np = (pred_binary.squeeze().cpu().numpy() * 255).astype('uint8')
    mask_img = Image.fromarray(mask_np)
    print(f"Mask size: {mask_np.shape}")
    
    print("Saving mask...")
    mask_img.save(output_path)
    print("Mask saved successfully")

    return output_path
