import torch
import torch.nn as nn
import numpy as np
import os
from helper import *
from models import TransformerEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(FOLDER_PATH, "plots")
MODEL_PATH = os.path.join(FOLDER_PATH, "models")
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Hyperparameters
N_epochs = 100
BATCH_SIZE = 128  # Batch size
LEARNING_RATE = 4e-4
d_model = 64
nhead = 2
dim_feedforward = 256
num_layers_encoder = 2




# loading data 
train_dataloader, val_dataloader, test_dataloader = get_dataloader(batch_size=BATCH_SIZE)

# initialize the model
model = TransformerEncoder(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    num_layers_encoder=num_layers_encoder,
    input_dim=3,
    output_dim=2
)

# define the loss function
criterion = nn.MSELoss()
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# define the device
device = 'mps'

model.to(device)

# training loop
epochs = tqdm(range(N_epochs), desc="Training", unit="epoch")
best_val_loss = float('inf')
all_train_losses = []
all_val_losses = []
plot_intvl = 1
for epoch in epochs:
    model.train()
    train_loss = 0.0
    i=0
    for data, target in train_dataloader:
        src, lengths = data
        data, target = [src.to(device),lengths], target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        i+=1

    train_loss /= i
    all_train_losses.append(train_loss)

    # validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        i=0
        for data, target in val_dataloader:
            src, lengths = data
            data, target = [src.to(device),lengths], target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            i+=1
    val_loss /= i
    all_val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{N_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # update the learning rate
    scheduler.step()
    # save the model if the validation loss is lower than the previous best
    if epoch == 0 or val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'{model.name}_best.pth'))
        print(f"Model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")

    # plot the training and validation loss
    # plot the validation loss and training loss every plot_intvl epochs
    if (epoch % plot_intvl == 0) or epoch==N_epochs-1:
        if epoch==N_epochs-1:
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

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(0,epoch+1,1),all_train_losses, label='Training Loss')
        plt.plot(np.arange(0,epoch+1,1),all_val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(PLOT_PATH, f"{model.name}_loss_plot.png"))
        plt.close()

