import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import argparse
from helper import *

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





FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER_PATH, "data")

# Hyperparameters
batch_size = 32
train_fraction = 0.7 # Fraction of the data used for training
val_fraction = 0.15 # Fraction of the data used for validation
k = 5 # Number of nearest neighbors to consider
output_dims = [12, 24, 12] # Output dimensions of the model

# Load the data
train_dataset, val_dataset, test_dataset, n_labels, label_ranges, data_ranges = get_normalized_data(DATA_PATH)
label_ranges = label_ranges["test"]
print(label_ranges)

# Load the data into DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_gnn)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn_gnn)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn_gnn)


print('###########################')
print('### Plotting the model ###')
print('###########################')

# Initialize model
model_choice = 'GNN'
model = initialize_model(model_choice, k=k, output_dims=output_dims)


# Load the best model saved in models directory
if os.path.exists(FOLDER_PATH+f'/models/{model.model_name}_best.pth'):
    best_model = torch.load(FOLDER_PATH+f'/models/{model.model_name}_best.pth', weights_only=True)
    print("Best model loaded from file.")
else:
    raise FileNotFoundError(f"Best model file not found in {FOLDER_PATH}/models/")

# Final evaluation on the test dataset
model.load_state_dict(best_model)
model.eval()
device = "cpu"
model.to(device)

all_predictions, all_true_labels, _, _ = evaluate_model(model, test_loader, loss_function, device)
all_predictions = denormalize(all_predictions[:,:n_labels], np.array([label_ranges['xpos'], label_ranges['ypos']]).T)
all_true_labels = denormalize(all_true_labels,  np.array([label_ranges['xpos'], label_ranges['ypos']]).T)

gt = all_true_labels # for readability

# Scatter plots for predictions
fig, axes = plt.subplots(1, 2, figsize=(16, 10))  # First row taller
labelNames = ['xpos', 'ypos']
for j in range(n_labels):
    # Scatter plot
    ax = axes[j]
    # Scatter plot of true vs predicted values with error bars
    ax.scatter(gt[:, j], all_predictions[:, j],marker='o', alpha=0.2)
    ax.plot([gt[:, j].min().item(), gt[:, j].max().item()], [gt[:, j].min().item(), gt[:, j].max().item()],
            c="black", linestyle="dashed", label="Perfect prediction")
    ax.set_xlabel("true " + labelNames[j])
    ax.set_ylabel("predicted " + labelNames[j])
    ax.legend()

plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_scatter.png')
plt.close()


# Plot the error distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 10))  # First row taller
for j in range(n_labels):
    ax = axes[j]
    # Calculate the error
    error = all_predictions[:, j] - gt[:, j]
    # Plot the error
    ax.hist(error, bins=50, density=True, alpha=0.7, color='xkcd:gray', label="Error distribution")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Error distribution for {labelNames[j]}")
    # Add a vertical line at zero
    ax.axvline(0, color='black', linestyle='dashed', linewidth=1, label='Zero error')
    # Add a gaussian fit
    mu, std = np.mean(error), np.std(error)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    ax.plot(x, p, 'xkcd:red', linewidth=2, label='Gaussian fit\n'+fr'$\mu={mu:.2f}$,'+'\n'+fr'$\sigma={std:.2f}$')
    # Add a legend
    ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_error_distribution.png')
plt.close()


# Plot the error as a function of the true value
fig, axes = plt.subplots(1,2, figsize=(16, 10))  # First row taller
for j in range(n_labels):
    ax = axes[j]
    # Calculate the error
    error = all_predictions[:, j] - gt[:, j]
    # Plot the error
    ax.scatter(gt[:, j], error, marker='o', alpha=0.2)
    ax.set_xlabel("true " + labelNames[j])
    ax.set_ylabel("Error")
    ax.set_title(f"Error as a function of true {labelNames[j]}")
    # Add a horizontal line at zero
    ax.axhline(0, color='black', linestyle='dashed', linewidth=1, label='Zero error')
    # Add a legend
    ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_error_vs_true.png')
plt.close()


# Plot the error as a function of the predicted value
fig, axes = plt.subplots(1, 2, figsize=(16, 10))  # First row taller
for j in range(n_labels):
    ax = axes[j]
    # Calculate the error
    error = all_predictions[:, j] - gt[:, j]
    # Plot the error
    ax.scatter(all_predictions[:, j], error, marker='o', alpha=0.2)
    ax.set_xlabel("predicted " + labelNames[j])
    ax.set_ylabel("Error")
    ax.set_title(f"Error as a function of predicted {labelNames[j]}")
    # Add a horizontal line at zero
    ax.axhline(0, color='black', linestyle='dashed', linewidth=1, label='Zero error')
    # Add a legend
    ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_error_vs_predicted.png')
plt.close()
