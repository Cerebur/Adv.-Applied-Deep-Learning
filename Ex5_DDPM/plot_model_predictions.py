from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns  # a useful plotting library on top of matplotlib
from models import NoisePredictor  # the neural network that predicts the noise added to the data
import os


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


# generate a dataset of 1D data from a mixture of two Gaussians
# this is a simple example, but you can use any distribution
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1, 2])),
    torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
)
dataset = data_distribution.sample(torch.Size([10000]))  # create training data set
dataset_validation = data_distribution.sample(torch.Size([1000])) # create validation data set


# we will keep these parameters fixed throughout
# these parameters should give you an acceptable result
# but feel free to play with them
TIME_STEPS = 250
BETA = 0.02
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4

# Initialize model
model = NoisePredictor(input_dim=2, output_dim=1, hidden_dim=64, num_layers=5)  # the neural network that predicts the noise added to the data


model_name='variable_beta'  # name of the model
# Load the best model saved in models directory
if os.path.exists(FOLDER_PATH+f'/models/{model_name}_best.pth'):
    best_model = torch.load(FOLDER_PATH+f'/models/{model_name}_best.pth')
    print("Best model loaded from file.")
else:
    raise FileNotFoundError(f"Best model file not found in {FOLDER_PATH}/models/")



# Final evaluation on the test dataset
model.load_state_dict(best_model)
model.eval()
device = 'mps'
model.to(device)

beta_schedule = torch.linspace(BETA,0.001, TIME_STEPS).to(device)  # Linearly decreasing beta values
if model_name == 'fixed_beta':
    beta_schedule = beta_schedule * 0 + BETA
alpha = 1 - beta_schedule
alpha_bar = torch.cumprod(alpha, dim=0)  # Cumulative product of (1 - beta_t)

def sample_reverse(g, count, samples_evolution_num=10, hist_sample_times=10):
    """
    Sample from the model by applying the reverse diffusion process

    Here, implement algorithm 2 of the DDPM paper (https://arxiv.org/abs/2006.11239)

    Parameters
    ----------
    g : torch.nn.Module
        The neural network that predicts the noise added to the data
    count : int
        The number of samples to generate in parallel

    Returns
    -------
    x : torch.Tensor
        The final sample from the model
    """
    
    # sample a random noise from the standard normal distribution
    x = torch.randn(count, 1).to(device)  # shape (count, 1)
    # randomly select 10 indices from x array to plot their evolution
    indices = torch.randint(0, count, (samples_evolution_num,))
    # save the evolution of the samples
    x_evolution = torch.zeros((samples_evolution_num, TIME_STEPS)).to(device)
    x_evolution[:, 0] = x[indices].squeeze(1)  # save the initial samples
    hist_tot=[]
    bins_tot = []
    # loop over time steps
    with torch.no_grad():
        for t in range(TIME_STEPS-1,-1,-1):
            # compute sqrt_alpha_bar_t and sqrt(1 - alpha_bar_t) for each t
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).to(device)
            # sample a random noise from the standard normal distribution
            z = torch.randn(count, 1).to(device) if t > 0 else torch.zeros_like(x).to(device)  # shape (count, 1)
            # compute the new sample
            x = 1 / torch.sqrt(alpha[t]) * (x - (1 - alpha[t]) / sqrt_one_minus_alpha_bar_t * g(x, torch.full((count,),t+1).to(device))) + z * torch.sqrt(beta_schedule[t])
            # save the evolution of the samples
            x_evolution[:, t] = x[indices].squeeze(1)  # save the current samples

            # calc hist of the samples every hist_sample_times steps
            if t % hist_sample_times == 0:
                # calculate the histogram of the samples
                hist, binedges = np.histogram(x.detach().cpu().numpy(), bins=50, range=(-10, 10), density=True)
                hist = hist / hist.sum()
                hist_tot.append(hist)
                bins= binedges[:-1] + np.diff(binedges) / 2
                bins_tot.append(bins)


    return x, x_evolution, hist_tot, bins_tot
sample_times=25
samples, samples_evolution, hist_tot, bins_tot = sample_reverse(model, 1000, samples_evolution_num=100, hist_sample_times=sample_times)
samples = samples.detach().cpu().numpy()

# plot the samples
fig, ax = plt.subplots(1, 1)
bins = np.linspace(-10, 10, 100)
sns.kdeplot(dataset, ax=ax, color='xkcd:red', label='True distribution', linewidth=2)
sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution', stat='density')
ax.legend()
ax.set_xlabel('Sample value')
ax.set_ylabel('Sample count')
fig.tight_layout()
plt.savefig(os.path.join(FOLDER_PATH, "plots", f"{model_name}_sampled_distribution.png"))
plt.close()

# plot the evolution of the samples
fig, ax = plt.subplots(1, 1)
for i in range(len(samples_evolution)):
    # check whether the sample ends up below 0 or above 0 and color accordingly
    if samples_evolution[i][0] < 0:
        color = 'xkcd:dirty blue'
    else:
        color = 'xkcd:pale orange'
    ax.plot(torch.arange(TIME_STEPS, 0, -1),samples_evolution[i].detach().cpu().numpy(), label=f'Sample {i+1}', color=color, alpha=0.5)
ax.set_xlabel('Time step')
ax.set_ylabel('Sample value')
fig.tight_layout()
plt.savefig(os.path.join(FOLDER_PATH, "plots", f"{model_name}_sample_evolution.png"))
plt.close()

# plot the histograms of the samples as one multi plot
fig, ax = plt.subplots(250//sample_times//5, 5, figsize=(20, 8))
for i in range(len(hist_tot)):
    a = i // 5
    b = i % 5

    ax[a, b].bar(bins_tot[i], hist_tot[i], facecolor='xkcd:greyblue', edgecolor='black', label=f'Time step {i*sample_times}', width=20/len(bins_tot[i]), alpha=0.7)
    ax[a, b].set_title(f'Time step {(i+1)*sample_times}')
    ax[a, b].set_xlabel('Sample value')
    ax[a, b].set_ylabel('Sample count')
fig.tight_layout()
plt.savefig(os.path.join(FOLDER_PATH, "plots", f"{model_name}_sample_histograms.png"))
plt.close()