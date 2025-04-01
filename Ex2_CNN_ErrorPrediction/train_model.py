import os
from torchinfo import summary
from helper import train_model, plot_fn, loss_function, prepare_data, initialize_model, get_device

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/Users/jonathan/Documents/Homeworks_with_Python/Adv. Deep Learning/Ex1_VanillaCNN/data/galah4"

# Hyperparameters
learning_rate = 2e-4
batch_size = 32
num_epochs = 100 # 100
patience = 10 # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7 # Fraction of the data used for training
val_fraction = 0.15 # Fraction of the data used for validation

# Prepare data
train_loader, val_loader, test_loader, spectra_length, n_labels, labelNames, ranges = prepare_data(
    DATA_PATH, train_fraction, val_fraction, batch_size
)

# Initialize model
model_choice = 'CNNstd'
model = initialize_model(model_choice, n_labels)

# Print the model summary before moving it to the device
summary(model, input_size=(1, 1, spectra_length))

# Detect and use the appropriate device
device = get_device()
print(f"Using device: {device}")
model.to(device)

# Train the model
train_losses, val_losses, best_model = train_model(
    model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience,
    device, model_name=model_choice, plot_fn=plot_fn, plot_kwargs={"plot_folder": FOLDER_PATH+'/plots'},
    plot_interval=10
)


