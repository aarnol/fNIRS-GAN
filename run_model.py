import torch
from trainer import CGANTrainer
from model import Generator, Discriminator
from dataset import *
from torch.utils.data import DataLoader


if __name__ == "__main__":
    

    # Hyperparameters
    latent_dim = 100
    num_classes = 2
    batch_size = 4
    g_lr = 0.0002
    d_lr = 0.0001
    beta1 = 0.5
    epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    # Initialize generator and discriminator
    generator = Generator(latent_dim, num_classes)
    discriminator = Discriminator(num_classes)

    # Initialize dataloader
    subject_ids = [1]
    dataset = ONRData("int")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    # Initialize trainer
    trainer = CGANTrainer(generator, discriminator, dataloader, device, g_lr, d_lr, beta1, epochs)
    
    # Train model
    trainer.train()

    # Save model
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Model trained and saved successfully!")

  