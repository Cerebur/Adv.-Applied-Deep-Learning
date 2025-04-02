import time
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import CNN_std, CNN_normalizingFlow

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

def normalize(labels, p):
    """
    Normalize the input labels using percentile-based scaling.

    This function scales the input labels to a range of [0, 1] based on the specified percentiles.
    The scaling is done by computing the percentiles of the labels and then normalizing the labels
    using these percentile values.

    Parameters:
    labels (np.ndarray): The input labels to be normalized.
    p (float): The percentile value used for scaling. The function uses the p-th and (1-p)-th percentiles
               for normalization.

    Returns:
    tuple: A tuple containing the normalized labels and the range used for normalization.
           - normalized_labels (np.ndarray): The normalized labels.
           - ranges (np.ndarray): The range used for normalization, which includes the p-th and (1-p)-th percentiles.
    """
    ranges = np.percentile(labels, [100 * p, 100 * (1 - p)], axis=0)
    labels = (labels - ranges[0]) / (ranges[1] - ranges[0])
    return labels, ranges

# Function to denormalize the labels back to their original scale
def denormalize(labels, ranges):
    """
    Denormalize the input labels using the specified range.

    This function denormalizes the input labels using the specified range values.
    The denormalization is done by scaling the labels back to the original range
    using the provided range values.

    Parameters:
        labels (np.ndarray): The normalized labels to be  denormalized.
        ranges (np.ndarray): The range values used for normalization.

    Returns:
        np.ndarray: The denormalized labels.
    """
    return labels * (ranges[1] - ranges[0]) + ranges[0]

def denormalize_std(uncertainty, ranges):
    """
    Denormalizes the given uncertainty predictions using the provided range.

    It is different to the denormalization of the labels which also includes a shift.

    Parameters
    ----------
    uncertainty : array-like
        The normalized uncertainty to be denormalized.
    ranges : array-like
        A two-element array-like object where the first element is the minimum value
        and the second element is the maximum value of the original range.
    Returns
    -------
    array-like
        The denormalized uncertainty.
    """

    return uncertainty * (ranges[1] - ranges[0])


def get_normalized_data(data_path, return_SNR=False):
    """
    Load and normalize spectra and label data from the given path.
    Parameters
    ----------
    data_path : str
        The path to the directory containing the spectra and labels data files.
    Returns
    -------
    spectra : numpy.ndarray
        The normalized spectra data.
    labels : numpy.ndarray
        The normalized labels data (t_eff, log_g, fe_h).
    spectra_length : int
        The length of the spectra.
    n_labels : int
        The number of labels used (should be 3).
    labelNames : list of str
        The names of the labels used (t_eff, log_g, fe_h).
    ranges : numpy.ndarray
        The ranges used for normalization of the labels.
    """

    # Load the spectra data
    spectra = np.load(f"{data_path}/spectra.npy")
    spectra_length = spectra.shape[1]

    # Load the labels data
    # labels: mass, age, l_bol, dist, t_eff, log_g, fe_h
    labelNames = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h"]
    labels = np.load(f"{data_path}/labels.npy")
    SNR = labels[:, -1]
    labels = labels[:, :-1]

    # We only use the labels: t_eff, log_g, fe_h
    labelNames = labelNames[-3:]
    labels = labels[:, -3:]
    n_labels = labels.shape[1]

    labels, ranges = normalize(labels, 0.05)

    # Normalize spectra
    spectra = np.log(np.maximum(spectra, 0.2))

    if return_SNR:
        return spectra, labels, spectra_length, n_labels, labelNames, ranges, SNR
    return spectra, labels, spectra_length, n_labels, labelNames, ranges

def mse_loss(predictions, labels):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and true labels.

    Parameters
    ----------
    predictions : array-like
        The predicted values.
    labels : array-like
        The true labels.
    Returns
    -------
    float
        The computed MSE loss.
    """
    return nn.MSELoss()(predictions, labels)

def nll_loss(predictions, batch_labels, n_labels):
    """
    Computes the Negative Log Likelihood (NLL) loss between predictions and true labels.

    Parameters
    ----------
    predictions : array-like
        The predicted values.
    batch_labels : array-like
        The true labels.
    n_labels : int
        The number of labels used in the model.
    Returns
    -------
    float
        The computed NLL loss.
    """
    mean = predictions[:, :n_labels]
    log_std = predictions[:, n_labels:]
    std = torch.exp(log_std)
    nll = (0.5 * ((batch_labels-mean)/std)**2) + log_std
    return nll.mean()

def nf_loss(inputs, batch_labels, model):
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(batch_labels, inputs) # get the probability of the labels given the input data
    loss = -log_pdfs.mean() # take the negative mean of the log probabilities
    return loss


def loss_function(inputs, labels, model, loss_type='nll'):
    """
    Computes the loss between the model predictions and the true labels using negative log-likelihood (NLL).

    Parameters
    ----------
    inputs : array-like
        The input data to the model.
    labels : array-like
        The true labels corresponding to the input data.
    model : object
        The model used to make predictions.

    Returns
    -------
    float
        The computed negative log-likelihood error loss.
    """

    if loss_type == 'normalizing_flow':
        # For normalizing flow models
        loss = nf_loss(inputs, labels, model)
        return loss

    predictions = model(inputs)

    loss_type_dict = {
        'nll': [nll_loss, [predictions, labels, 3]],
        'mse': [nn.MSELoss(),[ predictions, labels]]
    }

    if loss_type not in loss_type_dict:
        raise ValueError(f"Invalid loss type. Expected one of {list(loss_type_dict.keys())}.")
    loss_calc,input_params = loss_type_dict[loss_type]
    loss_result = loss_calc(*input_params)
    return loss_result


def train_model(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience,
                device, plot_fn=None, plot_interval=10, plot_kwargs=None, save_model_full=False):
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
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # set the learning rate to decrease on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    last_lr = learning_rate

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        start_time = time.time()  # Start the timer for this epoch

        # Training phase
        model.train()
        total_train_loss = 0.0
        for step, (batch_spectra, batch_labels) in enumerate(train_loader):
            batch_spectra, batch_labels = batch_spectra.to(device).unsqueeze(1), batch_labels.to(device)  # Add channel dimension for CNNs
            optimizer.zero_grad()

            loss=loss_function(batch_spectra, batch_labels, model, loss_type=model.loss_name)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Print progress every 10th step, updating the same line
            if (step + 1) % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                sys.stdout.flush()

        sys.stdout.write("\n")  # Move to the next line after the epoch

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_spectra, batch_labels in val_loader:
                batch_spectra, batch_labels = batch_spectra.to(device).unsqueeze(1), batch_labels.to(device)

                val_loss=loss_function(batch_spectra, batch_labels, model)

                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = copy.copy(model.state_dict())
            # Save the best model to the "models" directory
            if not os.path.exists(FOLDER_PATH+'/models'):
                os.makedirs(FOLDER_PATH+'/models')
            if model.model_name is not None:
                torch.save(best_model, FOLDER_PATH+f"/models/{model.model_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                if model.model_name is not None:
                    torch.save(best_model, FOLDER_PATH+f"/models/{model.model_name}_best.pth")
                if(plot_fn is not None):
                    assert(plot_kwargs is not None)
                    assert("plot_folder" in plot_kwargs)
                    plot_fn(model.model_name,
                        train_losses,
                        val_losses,
                        plot_folder=plot_kwargs["plot_folder"],
                        suffix="epoch_%.5d" % epoch)
                break
            
        # Save the model with all epochs to plot the training and validation loss later
        if model.model_name is not None and save_model_full:
            if not os.path.exists(FOLDER_PATH+'/models'):
                os.makedirs(FOLDER_PATH+'/models')
            torch.save(model.state_dict(), FOLDER_PATH+f"/models/{model.model_name}_epoch_{epoch}.pth")

        if(epoch%plot_interval==0) or (epoch==num_epochs-1):
            if(plot_fn is not None):
                assert(plot_kwargs is not None)
                assert("plot_folder" in plot_kwargs)

                plot_fn(model.model_name,
                        train_losses,
                        val_losses,
                        plot_folder=plot_kwargs["plot_folder"],
                        suffix="epoch_%.5d" % epoch)

        # Decrease learning rate on plateau
        scheduler.step(metrics=avg_val_loss)
        if scheduler.get_last_lr()[0] != last_lr:
            print("Learning rate changed to {scheduler.get_last_lr()[0]:.2e}")
            last_lr = scheduler.get_last_lr()[0]

    return train_losses, val_losses, best_model

def evaluate_model(model, test_loader, loss_function, device):
    """
    Evaluate the given model on the test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    loss_function : callable
        Loss function used to compute the loss.
    device : torch.device
        Device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns
    -------
    all_predictions : numpy.ndarray
        Array of denormalized predictions made by the model.
    all_true_labels : numpy.ndarray
        Array of denormalized true labels from the test dataset.
    """
    print("Evaluating model on the test dataset...")
    model.eval()
    total_test_loss = 0.0
    all_predictions = []
    all_true_labels = []

    first_batch_spectra=None
    first_batch_labels=None

    with torch.no_grad():
        for batch_index, (batch_spectra, batch_labels) in enumerate(test_loader):
            batch_spectra, batch_labels = batch_spectra.to(device).unsqueeze(1), batch_labels.to(device)
            predictions = model(batch_spectra)

            test_loss = loss_function(batch_spectra, batch_labels,model)

            total_test_loss += test_loss.item()
            all_predictions.append(predictions.cpu())
            all_true_labels.append(batch_labels.cpu())

            if(batch_index==0):
                first_batch_spectra=batch_spectra
                first_batch_labels=batch_labels



    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    return torch.cat(all_predictions).numpy(), torch.cat(all_true_labels).numpy(), first_batch_spectra, first_batch_labels

def plot_fn(model_name,train_losses, val_losses, plot_folder, suffix=""):
    """
    Plot model predictions and training/validation losses.

    Parameters
    ----------
    train_losses : list of float
        List of training losses for each epoch.
    val_losses : list of float
        List of validation losses for each epoch.
    plot_folder : str
        Folder to save the plots.
    suffix : str, optional
        Suffix for the plot filenames. Default is an empty string.
    """
    import matplotlib.pyplot as plt

    # Ensure the plot folder exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Plot training and validation losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(f"{plot_folder}/loss_plot_{model_name}_{suffix}.png")
    plt.close()

    # Clean up the folder by removing old plots of the same model
    files = os.listdir(plot_folder)
    for file in files:
        if file.startswith(f"loss_plot_{model_name}") and file != f"loss_plot_{model_name}_{suffix}.png":
            os.remove(os.path.join(plot_folder, file))

def prepare_data(data_path, train_fraction, val_fraction, batch_size, seed=42):
    """
    Prepares the data by loading, normalizing, and splitting it into train, validation, and test sets.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    train_fraction : float
        Fraction of data to use for training.
    val_fraction : float
        Fraction of data to use for validation.
    batch_size : int
        Batch size for DataLoaders.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    tuple
        DataLoaders for train, validation, and test sets, along with metadata.
    """
    spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(data_path)
    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    total_samples = len(spectra_tensor)
    train_size = int(train_fraction * total_samples)
    val_size = int(val_fraction * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(spectra_tensor, labels_tensor),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, spectra_length, n_labels, labelNames, ranges

def initialize_model(model_choice, n_labels,nf_type=''):
    """
    Initializes the model based on the given choice.

    Parameters
    ----------
    model_choice : str
        The name of the model to initialize.
    n_labels : int
        Number of output labels for the model.

    Returns
    -------
    torch.nn.Module
        The initialized model.
    """
    model_dict = {'CNNflow': CNN_normalizingFlow(encoder=CNN_std, nf_type=nf_type)}
    if model_choice in model_dict:
        return model_dict[model_choice]
    else:
        raise ValueError(f"Invalid model choice. Please select one of {model_dict.keys()}.")

def get_device():
    """
    Detects and returns the appropriate device for computation.

    Returns
    -------
    torch.device
        The device to use ('cuda', 'mps', or 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



