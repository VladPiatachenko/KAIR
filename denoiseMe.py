import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import models.basicblock as B

print(cv2.__version__)

class FDnCNN(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=64, nb=20, act_mode='R'):
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FDnCNN().to(device)
model.load_state_dict(torch.load('5000_G.pth', map_location=device))
model.eval()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((512, 640)),  # Adjust as needed
    transforms.ToTensor(),
    transforms.Normalize((0.4, 0.4, 0.4), (0.8, 0.8, 0.8))  # Adjust normalization
])

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the frames
cv2.namedWindow("Original vs Denoised", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transformations
    frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    # Add an extra channel filled with zeros
    noise_channel = torch.zeros((1, 1, frame_tensor.size(2), frame_tensor.size(3)), device=device)
    frame_tensor_with_noise = torch.cat((frame_tensor, noise_channel), dim=1)

    # Perform denoising
    with torch.no_grad():
        denoised_image = model(frame_tensor_with_noise)

    # Post-process the output
    denoised_image = denoised_image.squeeze(0).clamp(0, 1).cpu().numpy() * 255
    denoised_frame_bgr = cv2.cvtColor(denoised_image.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Resize denoised frame to match the original frame's dimensions
    denoised_frame_resized = cv2.resize(denoised_frame_bgr, (frame.shape[1], frame.shape[0]))

    # Concatenate original and denoised frames
    display_frame = np.concatenate((frame, denoised_frame_resized), axis=1)

    # Display the frames
    cv2.imshow("Original vs Denoised", display_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
