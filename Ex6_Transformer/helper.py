import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import awkward
import numpy as np

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
# the data is stored in the folder data of the directory Ex3_DGCNN, which is in the parent directory of Ex6_Transformer
DATA_PATH = os.path.join(FOLDER_PATH, "..", "Ex3_DGCNN", "data")

def collate_fn_transformer(batch):
    """
    Custom function that defines how batches are formed.

    To process the batch items that each have a different number of hits, it is efficient
    to first concatenate all the data into a single tensor and save the lengths of each
    individual event to be able to split the data again later.

    # F: input_dim, number of features (time, x, y)
    # N: number of hits (different for each event)
    # B: batch size

    The resulting 2D tensor has the shape (B x N, F) where B is the batch size, N is the total number of hits of all events
    in the batch, and F is the number of features (time, x, y).


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
    lengths=[]

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        lengths.append(tensordata.shape[0])

        data_list.append(tensordata)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor

    data_vec=torch.cat(data_list) # (B, N, F)  -> (BxN, F) where B is the batch size, N is the number of hits, and F is the number of features (time, x, y)

    ## return a list [datalist, lengths]
    return [data_vec, lengths], labels

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
    flattened_data = awkward.flatten(data, axis=None)
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

    print (len(train_dataset), len(val_dataset), len(test_dataset))

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
        #ranges_x = np.array([np.min(x), np.max(x)])
        norm_x, ranges_x = normalize(x, 0.01) # Normalize the x data using the 1st and 99th percentiles
        y = dataset["data"][:, 2:3, :]
        #ranges_y = np.array([np.min(y), np.max(y)])
        norm_y, ranges_y = normalize(y, 0.01) # Normalize the y data using the 1st and 99th percentiles

        # Concatenate the normalized data back together
        dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)

        mean_std_dict={
            "xpos": (np.mean(dataset["xpos"]), np.std(dataset["xpos"])),
            "ypos": (np.mean(dataset["ypos"]), np.std(dataset["ypos"]))
        }

        ranges_label_x = np.array([np.min(dataset["xpos"]), np.max(dataset["xpos"])])
        ranges_label_y = np.array([np.min(dataset["ypos"]), np.max(dataset["ypos"])])

        # Normalize labels (this can be done in-place), e.g. by
        dataset["xpos"] = (dataset["xpos"] - np.mean(dataset["xpos"])) / np.std(dataset["xpos"])
        dataset["ypos"] = (dataset["ypos"] - np.mean(dataset["ypos"])) / np.std(dataset["ypos"]) 


        norm_ranges_dict={
            "time": ranges_times,
            "x": ranges_x,
            "y": ranges_y,
            "x_label": ranges_label_x,
            "y_label": ranges_label_y
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


def get_dataloader(batch_size=32):
    """
    Get the dataloaders for training, validation, and test datasets.

    Parameters:
        batch_size (int): The size of each batch.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU.

    Returns:
        tuple: A tuple containing the training, validation, and test dataloaders.
    """
    train_dataset, val_dataset, test_dataset, n_labels, mean_std_dict, norm_ranges_dict = get_normalized_data(DATA_PATH)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn_transformer)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn_transformer)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn_transformer)
    

    # Save the normalization parameters to a file
    normalization_params = {
        "x_pos_mean_std_train": mean_std_dict["train"]["xpos"],
        "y_pos_mean_std_train": mean_std_dict["train"]["ypos"],
        "x_pos_mean_std_val": mean_std_dict["val"]["xpos"],
        "y_pos_mean_std_val": mean_std_dict["val"]["ypos"],
        "x_pos_mean_std_test": mean_std_dict["test"]["xpos"],
        "y_pos_mean_std_test": mean_std_dict["test"]["ypos"],
        "x_ranges_train": norm_ranges_dict["train"]["x"],
        "y_ranges_train": norm_ranges_dict["train"]["y"],
        "time_ranges_train": norm_ranges_dict["train"]["time"],
        "x_ranges_val": norm_ranges_dict["val"]["x"],
        "y_ranges_val": norm_ranges_dict["val"]["y"],
        "time_ranges_val": norm_ranges_dict["val"]["time"],
        "x_ranges_test": norm_ranges_dict["test"]["x"],
        "y_ranges_test": norm_ranges_dict["test"]["y"],
        "time_ranges_test": norm_ranges_dict["test"]["time"],
        "x_label_ranges_train": norm_ranges_dict["train"]["x_label"],
        "y_label_ranges_train": norm_ranges_dict["train"]["y_label"],
        "x_label_ranges_val": norm_ranges_dict["val"]["x_label"],
        "y_label_ranges_val": norm_ranges_dict["val"]["y_label"],
        "x_label_ranges_test": norm_ranges_dict["test"]["x_label"],
        "y_label_ranges_test": norm_ranges_dict["test"]["y_label"]
    }

    normalization_params_path = os.path.join(FOLDER_PATH, "normalization_params.txt")
    with open(normalization_params_path, "w") as f:
        for key, value in normalization_params.items():
            f.write(f"{key}: {value}\n")
    print(f"Normalization parameters saved to {normalization_params_path}")


    def print_data_info(train_dataset, val_dataset, test_dataset):
        # to get familiar with the dataset, let's inspect it.
        print(f"The training dataset contains {len(train_dataset)} events.")
        print(f"The validation dataset contains {len(val_dataset)} events.")
        print(f"The test dataset contains {len(test_dataset)} events.")
        print(f"The training dataset has the following columns: {train_dataset.fields}")
        print(f"The validation dataset has the following columns: {val_dataset.fields}")
        print(f"The test dataset has the following columns: {test_dataset.fields}")
        # print the first event of the training dataset
        print(f"The first event of the training dataset is: {train_dataset[0]}")

        # We are interested in the labels xpos and ypos. This is the position of the neutrino interaction that we want to predict.
        print(f"The first event of the training dataset has the following labels: {train_dataset['xpos'][0]}, {train_dataset['ypos'][0]}")
        # Awkward arrays also allow us to obtain the 'xpos' and 'ypos' label for all events in the dataset
        print(f"The first 10 labels of the training dataset are: {train_dataset['xpos'][:10]}, {train_dataset['ypos'][:10]}")

        # The data can be accessed by using the 'data' key.
        # The data is a 3D array with the first dimension being the number of events,
        # the second dimension being the the three features (time, x, y)
        # the third dimension being the number of hits, i.e., detected photons.
        print(f"The first event of the training dataset has {len(train_dataset['data'][0][0])} hits, i.e., detected photons.")
        # Let's loop over all hits and print the time, x, and y coordinates of the first event.
        for i in range(len(train_dataset['data'][0, 0])):
            print(f"Hit {i}: time = {train_dataset['data'][0,0,i]}, x = {train_dataset['data'][0,1, i]}, y = {train_dataset['data'][0,2,i]}")
        # To get all hit times of the first event, you can use the following code:
        print(f"The first event of the training dataset has the following hit times: {train_dataset['data'][0, 0]}")
        print(f"The first event of the training dataset has the following hit x positions: {train_dataset['data'][0, 1]}")
        print(f"The first event of the training dataset has the following hit y positions: {train_dataset['data'][0, 2]}")


    #print_data_info(train_dataset, val_dataset, test_dataset)

    return train_loader, val_loader, test_loader


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
        i = 0
        for batch_index, (batch_data, batch_labels) in enumerate(test_loader):
            batch_src, batch_lengths = batch_data
            batch_data, batch_labels = [batch_src.to(device),batch_lengths], batch_labels.to(device)
            predictions = model(batch_data)

            test_loss = loss_function(predictions, batch_labels)

            total_test_loss += test_loss.item()
            all_predictions.append(predictions.cpu())
            all_true_labels.append(batch_labels.cpu())

            if(batch_index==0):
                first_batch_predictions=predictions
                first_batch_labels=batch_labels
            i+=1



    avg_test_loss = total_test_loss / i
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    return torch.cat(all_predictions).numpy(), torch.cat(all_true_labels).numpy(), first_batch_predictions, first_batch_labels