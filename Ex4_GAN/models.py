from torch import nn

class Generator_large(nn.Module):
    """
    Generator Model
    """
    def __init__(self, latent_dimension):
        """
        Args:
            latent_dimension (int): The dimension of the latent space
        """
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dimension, 4*4*1024),  # Input layer
            nn.BatchNorm1d(4*4*1024),  # Batch normalization
            nn.ReLU(),  # ReLU activation function
        )

        self.deconv1=nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)  # Deconvolution layer
        self.batchnorm1=nn.BatchNorm2d(512)  # Batch normalization
        self.deconv2=nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Deconvolution layer
        self.batchnorm2=nn.BatchNorm2d(256)  # Batch normalization
        self.deconv3=nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Deconvolution layer
        self.batchnorm3=nn.BatchNorm2d(128)  # Batch normalization
        self.deconv4 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=19)  # Adjusted kernel_size and padding
        self.batchnorm4=nn.BatchNorm2d(1)  # Batch normalization
        self.relu=nn.ReLU()  # ReLU activation function
        self.tanh=nn.Tanh()  # Tanh activation function



    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.deconv1(x)
        x = self.batchnorm1(x)
        # print('Shape after layer 1: ',x.shape)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.batchnorm2(x)
        # print('Shape after layer 2: ',x.shape)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.batchnorm4(x)
        #x = nn.Flatten()(x)
        # The image needs to be reshaped into an array of shape (batch_size, 28*28)
        x = x.view(-1, 28* 28)
        x = self.tanh(x)
        return x
    


class Generator(nn.Module):
    def __init__(self, latent_dimension, image_dimension=784):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dimension, 128),  # Input layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, 256),  # Hidden layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(256, 512),  # Hidden layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, image_dimension),  # Output layer
            nn.Tanh()  # Tanh activation function to get the output in the range [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator_large(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self, image_dimension):
        """
        Args:
            image_dimension (int): The dimension of the input image
        """
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Convolutional layer
            nn.ReLU(),  # ReLU activation function
            nn.Flatten()  # Flatten the output
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 14 * 14, 512),  # Fully connected layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.3),  # Dropout layer to prevent overfitting
            nn.Linear(512, 1),  # Output layer, 2 classes (real or fake)
            nn.Sigmoid()  # Sigmoid activation function to get a probability
        )

        self.image_dimension = image_dimension

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Reshape input to (batch_size, 1, H, W)
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
    


class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self, image_dimension):
        """
        Args:
            image_dimension (int): The dimension of the input image
        """
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(image_dimension, 512),  # Fully connected layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.3),  # Dropout layer to prevent overfitting
            nn.Linear(512, 256),  # Hidden layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.3),  # Dropout layer to prevent overfitting
            nn.Linear(256, 1),  # Hidden layer
            nn.Sigmoid()  # Sigmoid activation function to get a probability
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return x