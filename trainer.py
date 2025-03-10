import torch
import os
import json
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class CGANTrainer:
    def __init__(self, generator, discriminator, dataloader, device, g_lr=0.0002, d_lr = 0.0002, beta1=0.5, epochs=100, checkpoint_dir='checkpoints'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(beta1, 0.999))
        
        self.fixed_noise = torch.randn(16, generator.latent_dim, device=device)
        self.fixed_labels = torch.randint(0, generator.num_classes, (16,), device=device)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.metrics = {"G_loss": [], "D_loss": []}  # Store losses
    
    def train(self):
        for epoch in range(self.epochs):
            epoch_G_loss = 0.0
            epoch_D_loss = 0.0
            num_batches = len(self.dataloader)
            
            for i, (real_images, labels) in enumerate(self.dataloader):
                for j in range(2):
                    batch_size = real_images.size(0)
                    real_images, labels = real_images.to(self.device), labels.to(self.device)
                    
                    real_targets = torch.ones(batch_size, 1, device=self.device)
                    fake_targets = torch.zeros(batch_size, 1, device=self.device)
                    
                    # Train Discriminator
                    
                    self.optimizer_D.zero_grad()
                    outputs_real = self.discriminator(real_images, labels)
                    loss_real = self.criterion(outputs_real, real_targets)
                    
                    noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
                    fake_images = self.generator(noise, labels)
                    outputs_fake = self.discriminator(fake_images.detach(), labels)
                    loss_fake = self.criterion(outputs_fake, fake_targets)
                    if j == 0:
                        loss_D = loss_real + loss_fake
                        loss_D.backward()
                        self.optimizer_D.step()
                    
                    # Train Generator
                    self.optimizer_G.zero_grad()
                    outputs_fake = self.discriminator(fake_images, labels)
                    loss_G = self.criterion(outputs_fake, real_targets)
                    loss_G.backward()
                    self.optimizer_G.step()
                    
                    # Track losses
                    if j == 0:
                        epoch_G_loss += loss_G.item()
                        epoch_D_loss += loss_D.item()

                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{num_batches}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

            # Store average losses for the epoch
            avg_G_loss = epoch_G_loss / num_batches
            avg_D_loss = epoch_D_loss / num_batches
            self.metrics["G_loss"].append(avg_G_loss)
            self.metrics["D_loss"].append(avg_D_loss)

            print(f"Epoch {epoch+1} Completed -> Avg. Loss D: {avg_D_loss:.4f}, Avg. Loss G: {avg_G_loss:.4f}")
            
            self.save_checkpoint(epoch)
            self.save_metrics()

    def save_checkpoint(self, epoch):
        torch.save({'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict()},
                   os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        print(f"Checkpoint saved for epoch {epoch}")

    def save_metrics(self):
        metrics_path = os.path.join(self.checkpoint_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print("Metrics saved.")

class CGANTester:
    def __init__(self, generator, device, checkpoint_path):
        self.device = device
        self.generator = generator.to(device)
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

    def generate_images(self, num_images=16):
        noise = torch.randn(num_images, self.generator.latent_dim, device=self.device)
        labels = torch.randint(0, self.generator.num_classes, (num_images,), device=self.device)
        
        with torch.no_grad():
            fake_images = self.generator(noise, labels)
        
        self.display_images(fake_images)
    
    def display_images(self, images):
        images = (images + 1) / 2  # Normalize to [0,1] for visualization
        grid = vutils.make_grid(images.cpu(), normalize=True, nrow=4)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
