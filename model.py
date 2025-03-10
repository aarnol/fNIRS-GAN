import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = label_dim
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        self.fc = nn.Linear(latent_dim + label_dim, 384)
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=1, padding=1),
            
            nn.Tanh()
        )
    
    def forward(self, z, labels):
       
        label_embedded = self.label_embedding(labels)
        x = torch.cat((z, label_embedded), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 32, 3, 4)  # Reshape to (batch_size, 32, 3, 4)
        x = self.deconv_layers(x)
        return x[:, :, :12, :8]  # Ensure exact output shape (4, 12, 8)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(130, 320),
            nn.LeakyReLU(0.2),
            nn.Linear(320, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        
        label_embedded = self.label_embedding(labels)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, label_embedded), dim=1)
        x = self.fc_layers(x)
        return x

# Model initialization
latent_dim = 100
label_dim = 10  # Example number of labels
generator = Generator(latent_dim, label_dim)
discriminator = Discriminator(label_dim)
