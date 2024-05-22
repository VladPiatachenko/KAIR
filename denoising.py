import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import models.basicblock as B

class FDnCNN(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=64, nb=20, act_mode='R'):
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

# Load the model
model = FDnCNN()
model.load_state_dict(torch.load('5000_G.pth'))
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3, 0.3, 0.3), (0.15, 0.1, 0.1))
])

# Load and preprocess the image
image = Image.open('test.jpg').convert('RGB')
image_tensor = transform(image)

# Add a fourth channel (e.g., noise map)
noise_channel = torch.zeros_like(image_tensor[0:1, :, :])
image_tensor = torch.cat((image_tensor, noise_channel), dim=0)
image_tensor = image_tensor.unsqueeze(0)

# Debug: Check shapes
print(f"Input image tensor shape (with noise channel): {image_tensor.shape}")

# Perform denoising
start_time = time.time()  # Start timing

with torch.no_grad():
    denoised_image = model(image_tensor)

end_time = time.time()  # End timing

# Calculate processing time
processing_time = end_time - start_time
print(f"Image processing time: {processing_time:.4f} seconds")

# Debug: Check output tensor shape
print(f"Denoised image tensor shape: {denoised_image.shape}")

# Visualize output tensor values before denormalization
denoised_image_np = denoised_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
plt.imshow(denoised_image_np)
plt.title("Output Tensor Values Before Denormalization")
plt.show()

# Post-process the output
denoised_image = denoised_image.squeeze(0)  # Remove batch dimension
denoised_image = denoised_image[:3, :, :]  # Keep only RGB channels

# Adjust brightness and contrast
denoised_image = denoised_image.mul(0.7).add(0.5).clamp(0, 1)

# Convert tensor to PIL image
denoised_image_pil = transforms.ToPILImage()(denoised_image)

# Save or display the denoised image
denoised_image_pil.save('denoised_image.jpg')
denoised_image_pil.show()

# Visualize final denoised image
denoised_image_np_final = denoised_image.permute(1, 2, 0).cpu().numpy()
plt.imshow(denoised_image_np_final)
plt.title("Final Denoised Image")
plt.show()
