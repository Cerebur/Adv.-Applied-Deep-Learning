# # This is a simple example of a diffusion model in 1D.
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns  # a useful plotting library on top of matplotlib
from tqdm.auto import tqdm # a nice progress bar
import os
from torch.utils.data import DataLoader, TensorDataset
# For image transforms
from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# Download the data from huggingface (https://huggingface.co/datasets/simbaswe/galah4/tree/main)
# Then, specify this directory here
FILE_PATH = FOLDER_PATH#'/content/drive/MyDrive/Colab Notebooks/Adv. Deep Learning'
PLOT_PATH = FILE_PATH+'/plots/'
MODEL_PATH = FILE_PATH+'/models/'


# Select the gpu device if available
if torch.cuda.is_available():
    device = torch.device("cuda")       #CUDA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        #Apple GPU
else:
    device = torch.device("cpu")        #if nothing is found use the CPU
print(f"Using device: {device}")


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


# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
for i in dataset:
    # check if the data is normalized between 0 and 1
    assert i[0].min() >= 0 and i[0].max() <= 1, "Data is not normalized between 0 and 1"
# let's create a dataloader to load the data in batches
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, download=False, transform=myTransforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model_name = 'model1'  # Name of the model
DIM = 32
DIM_MULTS = (1, 2, 5)
model = Unet(
    dim = DIM,
    dim_mults = DIM_MULTS,
    flash_attn = False,
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = TIME_STEPS,           # number of steps
    sampling_timesteps = SAMPLING_TIMESTEPS    # number of sampling timesteps (using ddim for faster inference [see ddim paper])
).to(device)  # move the model to the device



optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


model_save_counter = 0  # Counter to save the model 100 epochs after last save
best_loss = float('inf')  # Initialize best loss to infinity
all_train_losses = []  # List to store all training losses for plotting
epochs = range(N_EPOCHS)  # this makes a nice progress bar
plot_intvl=1
plot_out_intvl = 5
for e in epochs:  # loop over epochs
    model.train()
    train_loss=0
    batches = tqdm(train_loader, leave=False)  # this makes a nice progress bar for the batches
    for i,batch in enumerate(batches):  # Use DataLoader for batching
        batch = batch[0]
        loss = diffusion(batch.to(device))  # compute the loss
        loss.backward()  # backpropagation
        optim.step()  # update the weights
        optim.zero_grad()  # Clear the gradients
        train_loss += loss.item()  # accumulate the loss

        # Update the progress bar only once per epoch
        batches.set_postfix(loss=train_loss/i)

    # compute the average loss
    avg_train_loss = train_loss / len(train_loader)
    all_train_losses.append(avg_train_loss)  # Append the average loss to the list

    if avg_train_loss < best_loss and model_save_counter >= 0:
        best_loss = avg_train_loss
        # save the model
        print(f"Saving model at epoch {e} with loss: {avg_train_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}_best.pth"))
        print(f"Model saved with loss: {avg_train_loss:.4f}")
        model_save_counter = 0
    else:
        model_save_counter += 1


    # plot the current network output every plot_out_intvl
    if (e % plot_out_intvl == 0) or e==len(train_loader)-1:
        model.eval()
        with torch.no_grad():
            samples = diffusion.sample(batch_size=32)
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[i].cpu().numpy().squeeze(), cmap='gray')
                ax.axis('off')
            # save images
            plt.savefig(os.path.join(PLOT_PATH, f"{model_name}_epoch_{e}.png"))
            plt.close()


    # plot the validation loss and training loss every plot_intvl epochs
    if (e % plot_intvl == 0) or e==len(train_loader)-1:
        if e==N_EPOCHS-1:
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

        # plot the average loss of the last 10 epochs
        es = np.arange(10, len(all_train_losses)*10+1, 10)
        plt.figure(figsize=(10, 5))
        plt.plot(es,all_train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(PLOT_PATH, f"{model_name}_loss_plot.png"))
        plt.close()






