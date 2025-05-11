from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns  # a useful plotting library on top of matplotlib
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import imageio


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# Add the path to your LaTeX installation
os.environ["PATH"] += os.pathsep + "/usr/local/bin/pdflatex"  # Update this path to your LaTeX installation

# some styling for nice plots
fig_width_pt=347.5*1.6
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/(2.0)    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inlw=2ches
fig_height = fig_width*golden_mean      # height in inches
fig_size = [fig_width,fig_height]
preamble = r"\usepackage{amsmath}" + "\n" + r"\usepackage{amssymb}" + "\n" + r"\usepackage{siunitx}"
plt.rcParams['text.latex.preamble']=preamble
params = {  'text.usetex': True,
            'font.weight': 'bold',
            'axes.linewidth' : 1.5,
            'axes.labelsize': 21,
            'font.size': 20,
            'legend.fontsize': 20,
            'xtick.labelsize': 20,
            'ytick.direction':'in',
            'xtick.direction':'in',
            'ytick.labelsize': 20,
            'font.family' : 'lmodern',
            'figure.figsize': fig_size} 
plt.rcParams.update(params)

# Hyperparameters
LEARNING_RATE = 4e-4
BATCH_SIZE = 128  # Batch size
N_EPOCHS = 100
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 250
BETA = 0.02


# we define a tranform that converts the image to tensor
myTransforms = transforms.Compose([transforms.ToTensor()])

# Select the gpu device if available
if torch.cuda.is_available():
    device = torch.device("cuda")       #CUDA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        #Apple GPU
else:
    device = torch.device("cpu")        #if nothing is found use the CPU
print(f"Using device: {device}")

model_name = 'model1'  # Name of the model
DIM = 32
DIM_MULTS = (1, 2, 5)
model = Unet(
    dim = DIM,
    dim_mults = DIM_MULTS,
    flash_attn = False,
    channels = 1
)


# Load the best model saved in models directory
if os.path.exists(FOLDER_PATH+f'/models/{model_name}_cpu_best.pth'):
    best_model = torch.load(FOLDER_PATH+f'/models/{model_name}_cpu_best.pth')
    print("Best model loaded from file.")
else:
    raise FileNotFoundError(f"Best model file not found in {FOLDER_PATH}/models/")


# Final evaluation on the test dataset
# model = torch.load(FOLDER_PATH+f'/models/{model_name}_best.pth', map_location=device)
model.load_state_dict(best_model)
# model.eval()

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = TIME_STEPS,           # number of steps
    sampling_timesteps = SAMPLING_TIMESTEPS    # number of sampling timesteps (using ddim for faster inference [see ddim paper])
).to(device)  # move the model to the device


# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
test_dataset = datasets.MNIST(root='dataset/', train=False, download=False, transform=myTransforms)
for i in test_dataset:
    # check if the data is normalized between 0 and 1
    assert i[0].min() >= 0 and i[0].max() <= 1, "Data is not normalized between 0 and 1"
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# test the final loss on the test dataset
def test_loss(model, diffusion, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            x = x.to(device)
            loss = diffusion(x)
            total_loss += loss.item()
    return total_loss / len(test_loader)
test_loss_value = test_loss(model, diffusion, test_loader, device)
print(f"Test loss: {test_loss_value:.4f}")

def tile_images(images, grid_shape=None):
    """
    Tile a batch of images into a single image grid.
    images: numpy array of shape (N, H, W)
    grid_shape: tuple (rows, cols), optional. If None, will use square grid.
    Returns: tiled image of shape (rows*H, cols*W)
    """
    N, H, W = images.shape
    if grid_shape is None:
        rows = cols = int(np.ceil(np.sqrt(N)))
    else:
        rows, cols = grid_shape
    # Pad images if needed
    pad = rows * cols - N
    if pad > 0:
        images = np.pad(images, ((0, pad), (0, 0), (0, 0)), mode='constant')
    images = images.reshape(rows, cols, H, W)
    images = images.transpose(0, 2, 1, 3)
    tiled = images.reshape(rows * H, cols * W)
    return tiled

def create_gif(model, diffusion, device, num_samples=9, num_timesteps=250, save_path='diffusion_process.gif'):
    """
    Create a gif of the reverse diffusion process for a batch of images, tiled as a grid.
    """
    batch_size = num_samples
    images = torch.randn(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)  # Start from pure noise

    frames = []

    for t in tqdm(reversed(range(num_timesteps))):
        with torch.no_grad():
            pred_x0, _ = diffusion.p_sample(images, t)
        if t%10==0 or t<=20:
            img = pred_x0.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 1)
            img = (img * 255).astype(np.uint8)
            img = img.squeeze(-1)  # (N, H, W)
            tiled_img = tile_images(img)
            frames.append(tiled_img)
        images = pred_x0

    imageio.mimsave(save_path, frames, fps=10)
    print(f"Gif saved at {save_path}")

# Create the gif
create_gif(model, diffusion, device, num_samples=25, num_timesteps=TIME_STEPS, save_path=os.path.join(FOLDER_PATH,'plots','diffusion_process.gif'))