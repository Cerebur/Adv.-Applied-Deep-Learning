# This is a helper file for the GAN exercise

# For os methods
import os
# For image transforms
from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
# FOR DATA LOADER
from torch.utils.data import DataLoader
# For numpy
import numpy as np
# For Pytorch methods
import torch
# For Pytorch methods
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# For torchvision methods
import torchvision
# For tensorboard
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
# For progress bar
from tqdm import tqdm
# For copy
import copy
# For deleting directories
import shutil


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER_PATH, "data")

def get_data(batchSize):
    """
    Loads the MNIST dataset and returns a DataLoader for training or testing.
    Parameters
    ----------
    batchSize : int
        The number of samples per batch to load.
    val_split : float, optional
        The proportion of the dataset to use for validation (default is 0.2). 
        Currently unused in this function.
    Returns
    -------
    DataLoader
        A PyTorch DataLoader object for the MNIST dataset, either for training or testing.
    Notes
    -----
    - The images are transformed to tensors and normalized to have a range of [-1, 1].
    - The MNIST dataset is downloaded automatically if not already present in the specified path.
    """

    # we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
    # which will convert the image range from [0, 1] to [-1, 1]
    myTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    # the MNIST dataset is available through torchvision.datasets
    print("loading MNIST digits dataset")
    dataset = datasets.MNIST(root=DATA_PATH+'/', transform=myTransforms, download=True, train=True)
    

    # let's create a dataloader to load the data in batches
    train_loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    print("MNIST digits train dataset loaded")
    return train_loader


def clear_tensorboard_logs(model_type='GAN'):
    """
    Clears the TensorBoard logs by deleting the log directory.
    """
    log_dir = os.path.join(FOLDER_PATH, 'runs', model_type)
    # Check if the log directory exists and delete it
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Cleared TensorBoard logs at {log_dir}")


def train_model(discriminator, generator, train_loader, learning_rate, num_epochs, model_type,
                device, latent_dimension, image_dimension, criterion, logStep, batch_size):
    """
    Trains a given model using the provided training and validation data loaders, loss function, and optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        loss_function : torch.nn.Module
            Loss function to be used for training.
        learning_rate : float
            learning rate
        num_epochs : int
            Number of epochs to train the model.
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        device : torch.device
            Device on which to perform training (e.g., 'cpu' or 'cuda').
        plot_fn : callable, optional
            Function to plot the model predictions during training. Default is None.
        plot_interval : int, optional
            Interval at which to plot the model predictions during training. Default is 10.
        plot_kwargs : dict, optional
            Additional keyword arguments to be passed to the plot function. Default is None.
        model_name : str, optional
            Name of the model for saving the best model. Default is None.
            If provided, the best model will be saved to the "models" directory with the given name.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - train_losses (list of float): List of average training losses for each epoch.
            - val_losses (list of float): List of average validation losses for each epoch.
    """
    
    # Clear TensorBoard logs before starting training
    clear_tensorboard_logs(model_type)

    opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
    opt_generator = optim.Adam(generator.parameters(), lr=learning_rate)

    best_model = None
    fixed_noise = torch.randn(batch_size, latent_dimension).to(device)  # fixed noise for visualization

    # Training Loop
    print("Started Training and visualization...")
    loss_generator = None
    loss_discriminator = None
    step = 0
    for epoch in range(num_epochs):
        print()
        for batch_idx, (real, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Train Discriminator
            real = real.view(-1, image_dimension).to(device)

            noise = torch.randn(batch_size, latent_dimension).to(device)
            fake = generator(noise)

            # Discriminator loss
            loss_real = criterion(discriminator(real), torch.ones_like(discriminator(real)))
            loss_fake = criterion(discriminator(fake.detach()), torch.zeros_like(discriminator(fake)))
            loss_discriminator = (loss_real + loss_fake) / 2

            opt_discriminator.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            opt_discriminator.step()

            # Generator loss
            loss_generator = criterion(discriminator(fake), torch.ones_like(discriminator(fake)))

            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            # Log to TensorBoard
            if batch_idx % logStep == 0:
                with torch.no_grad():
                    fake_images = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    real_images = real.reshape(-1, 1, 28, 28)
                    imgGridFake = torchvision.utils.make_grid(fake_images, normalize=True)
                    imgGridReal = torchvision.utils.make_grid(real_images, normalize=True)

                    writer = SummaryWriter(os.path.join(FOLDER_PATH, 'runs', 'GAN'))
                    writer.add_image("Fake Images", imgGridFake, step)
                    writer.add_image("Real Images", imgGridReal, step)
                    writer.add_scalar("Loss Discriminator", loss_discriminator, step)
                    writer.add_scalar("Loss Generator", loss_generator, step)

                step += 1

        print(f"\rEpoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} \ Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")

        # # Decrease learning rate on plateau
        # scheduler_discriminator.step(metrics=loss_discriminator)
        # if scheduler_discriminator.get_last_lr()[0] != lastlr_discriminator:
        #     print("Discriminator learning rate changed to {scheduler.get_last_lr()[0]:.2e}")
        #     lastlr_discriminator = scheduler_discriminator.get_last_lr()[0]
        # scheduler_generator.step(metrics=loss_generator)
        # if scheduler_generator.get_last_lr()[0] != lastlr_generator:
        #     print("Generator learning rate changed to {scheduler.get_last_lr()[0]:.2e}")
        #     lastlr_generator = scheduler_generator.get_last_lr()[0]
        # # Save the model if the validation loss is lower than the best validation loss
        # if save_model_full:
        #     avg_train_loss = (loss_discriminator.item()+loss_generator.item())/2
        #     if avg_train_loss < best_train_loss:
        #         best_train_loss = loss_generator
        #         patience_counter = 0
        #         best_model_generator = copy.deepcopy(generator.state_dict())
        #         best_model_discriminator = copy.deepcopy(discriminator.state_dict())
        #         best_model = {
        #             'generator': best_model_generator,
        #             'discriminator': best_model_discriminator
        #         }
        #         # Save the best model
        #         if not os.path.exists(os.path.join(FOLDER_PATH, 'models')):
        #             os.makedirs(os.path.join(FOLDER_PATH, 'models'))
        #         torch.save(best_model, os.path.join(FOLDER_PATH, 'models', f'GAN_best.pth'))
        #         print(f"Best model saved with validation loss: {best_train_loss:.4f}")
        #     else:
        #         patience_counter += 1
        #         if patience_counter >= patience:
        #             print(f"Early stopping triggered after {patience} epochs without improvement.")
        #             break


    return best_model



