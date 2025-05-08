# # This is a simple example of a diffusion model in 1D.
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns  # a useful plotting library on top of matplotlib
from tqdm.auto import tqdm # a nice progress bar
from models import NoisePredictor # the neural network that predicts the noise added to the data
import os
from torch.utils.data import DataLoader, TensorDataset




FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))


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
device= "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = 'variable_beta'  # Name of the model
g = NoisePredictor(input_dim=2,output_dim=1, name=model_name, hidden_dim=64, num_layers=5).to(device) # the neural network that predicts the noise added to the data


optimizer = torch.optim.Adam(g.parameters(), lr=LEARNING_RATE) # the optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Create a DataLoader for efficient batching and shuffling
train_loader = DataLoader(TensorDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)
# Create a DataLoader for validation
validation_loader = DataLoader(TensorDataset(dataset_validation), batch_size=BATCH_SIZE, shuffle=True)

# Define a beta schedule
beta_schedule = torch.linspace(0.001, BETA, TIME_STEPS).to(device)  # Linearly increasing beta values
if model_name == 'fixed_beta':
    beta_schedule = beta_schedule*0+BETA
alpha = 1 - beta_schedule
alpha_bar = torch.cumprod(alpha, dim=0)  # Cumulative product of (1 - beta_t)


model_save_counter = 0  # Counter to save the model 100 epochs after last save
best_loss = float('inf')  # Initialize best loss to infinity
all_val_losses = []  # List to store all losses for plotting
all_train_losses = []  # List to store all training losses for plotting
epochs = tqdm(range(N_EPOCHS))  # this makes a nice progress bar
for e in epochs:  # loop over epochs
    g.train()
    train_loss=0
    for batch in train_loader:  # Use DataLoader for batching
        x0 = batch[0].to(device)
        # sample a random time step t
        t = torch.randint(1, TIME_STEPS+1, (x0.shape[0],)).to(device)
        # sample a random noise
        noise = torch.randn_like(x0).to(x0.device)
        # compute sqrt_alpha_bar_t and sqrt(1 - alpha_bar_t) for each t
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t-1]).to(device)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t-1]).to(device)
        # add noise to the data
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        # compute the predicted noise
        predicted_noise = g(x_t, t)
        # compute the loss
        loss = torch.nn.MSELoss()(predicted_noise, noise.view(-1, 1))  # Ensure noise is reshaped correctly

        # backpropagation
        g.zero_grad()  # Clear the gradients
        loss.backward()
        # update the weights
        optimizer.step()
        train_loss += loss.item()
    # compute the average loss
    avg_train_loss = train_loss / len(train_loader)
    if not e==0:
        all_train_losses.append(avg_train_loss)  # Append the average loss to the list

    # validate the model
    g.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in validation_loader:  # Use DataLoader for batching
            x0 = batch[0].to(device)
            # sample a random time step t
            t = torch.randint(1, TIME_STEPS, (x0.shape[0],)).to(device)
            # sample a random noise
            noise = torch.randn_like(x0).to(x0.device)
            # compute sqrt_alpha_bar_t and sqrt(1 - alpha_bar_t) for each t
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).to(device)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).to(device)

            # add noise to the data
            x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

            # compute the predicted noise
            predicted_noise = g(x_t, t)
            # compute the loss
            loss = torch.nn.MSELoss()(predicted_noise, noise.view(-1, 1))  # Reshape noise to match predicted_noise

            total_loss += loss.item()
    # compute the average loss
    avg_loss = total_loss / len(validation_loader)
    if not e==0:
        all_val_losses.append(avg_loss)  # Append the average loss to the list
    if avg_loss < best_loss and model_save_counter >= 100:
        best_loss = avg_loss
        # save the model
        print(f"Saving model at epoch {e} with loss: {avg_loss:.4f}")
        if not os.path.exists(os.path.join(FOLDER_PATH, "models")):
            os.makedirs(os.path.join(FOLDER_PATH, "models"))
        torch.save(g.state_dict(), os.path.join(FOLDER_PATH, "models", f"{g.name}_best.pth"))
        print(f"Model saved with loss: {avg_loss:.4f}")
        model_save_counter = 0
    else:
        model_save_counter += 1


    # update the learning rate
    scheduler.step(avg_loss)


    # Update the progress bar only once per epoch
    epochs.set_postfix(loss=avg_loss)

    # plot the validation loss and training loss every 10 epochs
    if (e % 10 == 0 and e!=0) or e==N_EPOCHS-1:
        if not os.path.exists(os.path.join(FOLDER_PATH, "plots")):
            os.makedirs(os.path.join(FOLDER_PATH, "plots"))  # Ensure plots directory exists
        if e==N_EPOCHS-1:
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

        # plot the average loss of the last 10 epochs
        avg_train_losses = [np.mean(all_train_losses[i:i+10]) for i in range(0,len(all_train_losses),10)]
        avg_val_losses = [np.mean(all_val_losses[i:i+10]) for i in range(0,len(all_val_losses),10)]
        es = np.arange(10, len(avg_train_losses)*10+1, 10)
        plt.figure(figsize=(10, 5))
        plt.plot(es,avg_train_losses, label='Training Loss')
        plt.plot(es,avg_val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(FOLDER_PATH, "plots", f"{g.name}_loss_plot.png"))
        plt.close()

    

fig, ax = plt.subplots(1, 1)
sns.histplot(dataset, stat='density', label='Sampled distribution')
#plt.hist(dataset.numpy(), bins=50, density=True, alpha=0.5, color='blue', label='True distribution')
# plot the theoretical distribution
bins = np.linspace(-10, 10, 1000)
multi_gaussian = 1/np.sqrt(2 * np.pi) * (1/3*np.exp(-0.5 * (bins + 4)**2) + 2/3*np.exp(-0.5 * (bins - 4)**2))
ax.plot(bins, multi_gaussian, color='red', label='True distribution', linewidth=2)
plt.legend()
# save the plot in the plots folder and create the folder if it does not exist
if not os.path.exists(os.path.join(FOLDER_PATH, "plots")):
    os.makedirs(os.path.join(FOLDER_PATH, "plots"))
plt.savefig(os.path.join(FOLDER_PATH, "plots", "data_distribution.png"))
plt.close()



