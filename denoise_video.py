import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
import models.basicblock as B
from tqdm import tqdm

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FDnCNN().to(device)
model.load_state_dict(torch.load('5000_G.pth', map_location=device))
model.eval()

# Define video file path
video_path = 'vid_61a5aca9.mp4'

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Get total frames in the video
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.3, 0.3, 0.3), (0.15, 0.1, 0.1))
])

# Define codec and VideoWriter object
codec = cv2.VideoWriter_fourcc(*'MP4V')
output_video = cv2.VideoWriter('denoised_video.mp4', codec, 30.0, (640, 480))

# Initialize variables for time tracking
frame_times = []
start_time = time.time()

# Process frames from the video in batches
batch_size = 16
frames_buffer = []

for frame_idx in tqdm(range(total_frames)):
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb).to(device)
    noise_channel = torch.zeros_like(frame_tensor[0:1, :, :])
    image_tensor = torch.cat((frame_tensor, noise_channel), dim=0)
    frames_buffer.append(image_tensor)

    if len(frames_buffer) == batch_size or frame_idx == total_frames - 1:
        batch = torch.cat(frames_buffer, dim=0)
        frames_buffer = []

        with torch.no_grad():
            denoised_images = model(batch)

        denoised_images = denoised_images.clamp(0, 1).cpu().numpy() * 255
        denoised_images_bgr = [cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for img in denoised_images]

        for denoised_frame in denoised_images_bgr:
            output_video.write(denoised_frame)

end_time = time.time()
total_processing_time = end_time - start_time

video_capture.release()
output_video.release()

average_frame_time = total_processing_time / total_frames
print(f"Average Frame Processing Time: {average_frame_time:.4f} seconds")
print(f"Total Video Processing Time: {total_processing_time:.4f} seconds")
print("Video processing completed.")
