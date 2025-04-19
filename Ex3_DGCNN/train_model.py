import os
from helper import *
from torchinfo import summary
from torch.utils.data import DataLoader

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER_PATH, "data")

# Hyperparameters
learning_rate = 0.8e-4#2e-4
batch_size = 32
num_epochs = 100 # 100
patience = 40 # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7 # Fraction of the data used for training
val_fraction = 0.15 # Fraction of the data used for validation
k = 5 # Number of nearest neighbors to consider
output_dims = [12, 24, 12] # Output dimensions of the model

# Load the data
train_dataset, val_dataset, test_dataset, n_labels, _, _ = get_normalized_data(DATA_PATH)

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


# print_data_info(train_dataset, val_dataset, test_dataset)

print('###########################')
print('### Training the model ###')
print('###########################')
model_choice = 'GNN'
model = initialize_model(model_choice, k=k, output_dims=output_dims)

# Detect and use the appropriate device
device = 'cpu'
print(f"Using device: {device}")
model.to(device)


# summary(model, input_size=(batch_size, 3, 1000))

# Train the model
train_losses, val_losses, best_model = train_model(
    model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience,
    device, plot_fn=plot_fn, plot_kwargs={"plot_folder": FOLDER_PATH+'/plots'},
    plot_interval=10
)



