import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchinfo import summary
from helper import denormalize, denormalize_std, evaluate_model, loss_function, prepare_data, initialize_model, get_device


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/Users/jonathan/Documents/Homeworks_with_Python/Adv. Deep Learning/Ex1_VanillaCNN/data/galah4"

# Hyperparameters
batch_size = 32
train_fraction = 0.7 # Fraction of the data used for training
val_fraction = 0.15 # Fraction of the data used for validation

# Prepare data
_, _, test_loader, spectra_length, n_labels, labelNames, ranges = prepare_data(
    DATA_PATH, train_fraction, val_fraction, batch_size
)

# Initialize model
model_choice = 'CNNstd'
model = initialize_model(model_choice, n_labels)

# Print the model summary
summary(model, input_size=(1, 1, spectra_length))

# Load the best model saved in models directory
if os.path.exists(FOLDER_PATH+f'/models/{model_choice}_best.pth'):
    best_model = torch.load(FOLDER_PATH+f'/models/{model_choice}_best.pth')
    print("Best model loaded from file.")
else:
    raise FileNotFoundError(f"Best model file not found in {FOLDER_PATH}/models/")

# Final evaluation on the test dataset
model.load_state_dict(best_model)
model.eval()
device = get_device()
model.to(device)

all_predictions, all_true_labels, _, _ = evaluate_model(model, test_loader, loss_function, device)
all_uncertainties = np.exp(all_predictions[:, n_labels:])  # Assuming the last n_labels are uncertainties
all_predictions = denormalize(all_predictions[:,:n_labels], ranges)
all_uncertainties = denormalize_std(all_uncertainties, ranges)
all_true_labels = denormalize(all_true_labels, ranges)

# Scatter plots for predictions
fig, axes = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})  # First row taller
for j in range(n_labels):
    # Scatter plot
    ax = axes[0, j]
    gt = all_true_labels
    # Scatter plot of true vs predicted values with error bars
    ax.errorbar(gt[:, j], all_predictions[:, j], yerr=all_uncertainties[:, j], fmt='o', alpha=0.2, capsize=3)
    ax.plot([gt[:, j].min().item(), gt[:, j].max().item()], [gt[:, j].min().item(), gt[:, j].max().item()],
            c="black", linestyle="dashed", label="Perfect prediction")
    ax.set_xlabel("true " + labelNames[j])
    ax.set_ylabel("predicted " + labelNames[j])
    ax.legend()

    # Error plot
    ax = axes[1, j]
    errors = all_predictions[:, j] - gt[:, j]
    # Scatter plot of errors with respect to true values with error bars
    ax.errorbar(gt[:, j], errors, yerr=all_uncertainties[:, j], fmt='o', alpha=0.2, capsize=3, c="xkcd:red")
    #ax.scatter(gt[:, j], errors, s=6, alpha=0.2, c="red")
    ax.axhline(0, color="black", linestyle="dashed", label="Zero error")
    ax.set_xlabel("true " + labelNames[j])
    ax.set_ylabel("error (predicted - true)")
    ax.legend()

plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_scatter_and_error.png')
plt.close()



# Plot the pull distribution
pulls = (all_predictions - all_true_labels) / all_uncertainties
fig, axes = plt.subplots(1, 3, figsize=(16, 10))  # First row taller
for j in range(n_labels):
    ax = axes[j]
    ax.hist(pulls[:, j], bins=50, density=True, alpha=0.5)
    ax.axvline(0, color="black", linestyle="dashed", label="Zero pull")
    # Plot the Gaussian distribution
    mu, std = np.mean(pulls[:, j]), np.std(pulls[:, j])
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    p = np.exp(-0.5 * ((x - mu) / std)**2) / (std * np.sqrt(2 * np.pi))
    ax.plot(x, p, color='red', label=f'\nGaussian fit\n$\mu={mu:.2f}$\n$\sigma={std:.2f}$')
    ax.set_xlim(mu - 4*std, mu + 4*std)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("pull " + labelNames[j])
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_pull_distribution.png')
plt.close()

# Plot the uncertainty distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 10)) 
for j in range(n_labels):
    ax = axes[j]
    ax.hist(all_uncertainties[:, j], bins=50, density=True, alpha=0.5)
    ax.set_xlabel("uncertainty " + labelNames[j])
    ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig(FOLDER_PATH+f'/plots/{model_choice}_uncertainty_distribution.png')
plt.close()
