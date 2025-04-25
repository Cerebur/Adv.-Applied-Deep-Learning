import torchvision
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
# For os methods
import os
# For helper functions
from helper import *
# for the models
from models import Discriminator, Generator_large, Generator, Discriminator_large

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER_PATH, "data")

# Hyperparameters
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
lr = 3e-4
batchSize = 32  # Batch size
numEpochs = 100
model_type = 'GAN'  # 'GAN' or 'GAN_large'
logStep = 20 if model_type=='GAN_large' else 625  # the number of steps to log the images and losses to tensorboard
patience = 5 # number of epochs to wait before decreasing the learning rate if no improvement

latent_dimension = 128 # 64, 128, 256
# for simplicity we will flatten the image to a vector and to use simple MLP networks
# 28 * 28 * 1 flattens to 784
# you are also free to use CNNs
image_dimension = 28 * 28 * 1  # 784


# loading the data
train_loader = get_data(batchSize)

models={'GAN':[Discriminator,Generator],
        'GAN_large':[Discriminator_large,Generator_large]}


# initialize networks and optimizers
discriminator = models[model_type][0](image_dimension=image_dimension).to(device)
generator = models[model_type][1](latent_dimension=latent_dimension).to(device)
# This is a binary classification task, so we use Binary Cross Entropy Loss
criterion = nn.BCELoss()

# train the model
train_model(discriminator=discriminator,generator=generator,
            train_loader=train_loader,
            num_epochs=numEpochs,
            latent_dimension=latent_dimension,
            image_dimension=image_dimension,
            learning_rate=lr,
            model_type=model_type,
            device=device,
            criterion=criterion,
            logStep=logStep,
            batch_size=batchSize)
