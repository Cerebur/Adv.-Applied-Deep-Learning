import time
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import awkward
from torch_geometric.data import Data, Batch
from models import GNNEncoder
import awkward as ak

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER_PATH, "data")

def normalize(data, p):
    """
    Normalize the input data using percentile-based scaling.

    This function scales the input data to a range of [0, 1] based on the specified percentiles.
    The scaling is done by computing the percentiles of the data and then normalizing the data
    using these percentile values.

    Parameters:
    data (awkward.Array): The input data to be normalized.
    p (float): The percentile value used for scaling. The function uses the p-th and (1-p)-th percentiles
               for normalization.

    Returns:
    tuple: A tuple containing the normalized data and the range used for normalization.
           - normalized_data (awkward.Array): The normalized data.
           - ranges (tuple): The range used for normalization, which includes the p-th and (1-p)-th percentiles.
    """
    # Flatten the awkward array to compute percentiles
    flattened_data = ak.flatten(data, axis=None)
    lower_percentile = np.percentile(flattened_data, 100 * p)
    upper_percentile = np.percentile(flattened_data, 100 * (1 - p))

    # Normalize the data
    normalized_data = (data - lower_percentile) / (upper_percentile - lower_percentile)

    return normalized_data, (lower_percentile, upper_percentile)

# Function to denormalize the data back to their original scale
def denormalize(data, ranges):
    """
    Denormalize the input data using the specified range.

    This function denormalizes the input data using the specified range values.
    The denormalization is done by scaling the data back to the original range
    using the provided range values.

    Parameters:
        data (np.ndarray): The normalized data to be  denormalized.
        ranges (np.ndarray): The range values used for normalization.

    Returns:
        np.ndarray: The denormalized data.
    """
    return data * (ranges[1]) + ranges[0] # data * (ranges[1] - ranges[0]) + ranges[0]


def get_normalized_data(DATA_PATH):
    """Load, normalize, and return training, validation, and test datasets along with normalization parameters.
    This function reads datasets from parquet files located in the specified directory, normalizes the 
    time, x, and y coordinates in the 'data' field, and standardizes the 'xpos' and 'ypos' labels. 
    It also computes and returns the normalization parameters for the training dataset.
    DATA_PATH : str
        The path to the directory containing the 'train.pq', 'val.pq', and 'test.pq' parquet files.
    train_dataset : awkward.Array
        The normalized training dataset.
    val_dataset : awkward.Array
        The normalized validation dataset.
    test_dataset : awkward.Array
        The normalized test dataset.
    n_labels : int
        The number of labels in the dataset (2 for 'xpos', 'ypos').
    normalization_params : dict
        A dictionary containing the normalization parameters for the training dataset:
        - "x_mean_train": Mean of 'xpos' in the training dataset.
        - "x_std_train": Standard deviation of 'xpos' in the training dataset.
        - "y_mean_train": Mean of 'ypos' in the training dataset.
        - "y_std_train": Standard deviation of 'ypos' in the training dataset.
    Notes
    -----
    - The 'data' field in the datasets is normalized using the 1st and 99th percentiles for time, x, and y coordinates.
    - The 'xpos' and 'ypos' labels are standardized using the mean and standard deviation of the training dataset.
    - Awkward arrays are used for handling the datasets, and care is taken to preserve the dimensionality during normalization.
    """

    # Load the dataset
    train_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "train.pq"))
    val_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "val.pq"))
    test_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "test.pq"))

    #print the minimum and maximum values of the training dataset labels xpos and ypos
    print(f"Minimum and maximum values of the test dataset labels xpos: {np.min(test_dataset['xpos'])}, {np.max(test_dataset['xpos'])}")
    print(f"Minimum and maximum values of the test dataset labels ypos: {np.min(test_dataset['ypos'])}, {np.max(test_dataset['ypos'])}")

    # Normalize data and labels
    # working with Awkward arrays is a bit tricky because the ['data'] field can't be assigned in-place,
    # so we need to extract the time, x, and y coordinates, normalize them separately,
    # and then concatenate them back together.
    def normalize_dataset(dataset):
        times = dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                                    # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
        norm_times, ranges_times = normalize(times, 0.01) # Normalize the time data using the 1st and 99th percentiles
        x = dataset["data"][:, 1:2, :]
        norm_x, ranges_x = normalize(x, 0.01) # Normalize the x data using the 1st and 99th percentiles
        y = dataset["data"][:, 2:3, :]
        norm_y, ranges_y = normalize(y, 0.01) # Normalize the y data using the 1st and 99th percentiles

        # Concatenate the normalized data back together
        dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)

        mean_std_dict={
            "xpos": (np.mean(dataset["xpos"]), np.std(dataset["xpos"])),
            "ypos": (np.mean(dataset["ypos"]), np.std(dataset["ypos"]))
        }
        # Normalize labels (this can be done in-place), e.g. by
        dataset["xpos"] = (dataset["xpos"] - np.mean(dataset["xpos"])) / np.std(dataset["xpos"])
        dataset["ypos"] = (dataset["ypos"] - np.mean(dataset["ypos"])) / np.std(dataset["ypos"]) 


        norm_ranges_dict={
            "time": ranges_times,
            "x": ranges_x,
            "y": ranges_y
        }

        return dataset, mean_std_dict, norm_ranges_dict
    

    train_dataset, mean_std_dict_train, norm_ranges_dict_train = normalize_dataset(train_dataset)
    val_dataset, mean_std_dict_val, norm_ranges_dict_val = normalize_dataset(val_dataset)
    test_dataset, mean_std_dict_test, norm_ranges_dict_test = normalize_dataset(test_dataset)

    n_labels = 2  # Number of labels (xpos and ypos)

    mean_std_dict={
        "train": mean_std_dict_train,
        "val": mean_std_dict_val,
        "test": mean_std_dict_test
    }

    norm_ranges_dict={
        "train": norm_ranges_dict_train,
        "val": norm_ranges_dict_val,
        "test": norm_ranges_dict_test
    }

    return train_dataset, val_dataset, test_dataset, n_labels, mean_std_dict, norm_ranges_dict


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


def loss_function(inputs, labels, model, loss_type='mse'):
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

    predictions = model(inputs)

    loss_type_dict = {
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
        for step, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            loss=loss_function(data, labels, model)

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
            for data, labels in val_loader:
                val_loss=loss_function(data, labels, model)
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

    first_batch_predictions=None
    first_batch_labels=None

    with torch.no_grad():
        for batch_index, (batch_predictions, batch_labels) in enumerate(test_loader):
            batch_predictions, batch_labels = batch_predictions.to(device), batch_labels.to(device)
            predictions = model(batch_predictions)

            test_loss = loss_function(batch_predictions, batch_labels,model)

            total_test_loss += test_loss.item()
            all_predictions.append(predictions.cpu())
            all_true_labels.append(batch_labels.cpu())

            if(batch_index==0):
                first_batch_predictions=batch_predictions
                first_batch_labels=batch_labels



    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    return torch.cat(all_predictions).numpy(), torch.cat(all_true_labels).numpy(), first_batch_predictions, first_batch_labels

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



def initialize_model(model_choice, k, input_dim=3, output_dims=[12, 24, 12], num_targets=2):
    """
    Initializes the model based on the given choice.

    Parameters
    ----------
    model_choice : str
        The name of the model to initialize.
    k : int
        Number of nearest neighbors to consider.
    input_dim : int
        Number of features in the input data.
    output_dims : list
        List of output dimensions for each layer.
    num_targets : int
        Number of target dimensions.

    Returns
    -------
    torch.nn.Module
        The initialized model.
    """
    model_dict = {'GNN': GNNEncoder(k, input_dim, output_dims, num_targets)}
    if model_choice in model_dict:
        return model_dict[model_choice]
    else:
        raise ValueError(f"Invalid model choice. Please select one of {model_dict.keys()}.")


# Create the DataLoader for training, validation, and test datasets
# Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)
def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensordata)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list) # convert the list of Data objects to a single Batch object
    return packed_data, labels