import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import traceback

# Assuming these are imported from your custom files
from dataset import ONRData
from diffusion import UNet

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_schedule='cosine', device=None):
        """
        Initialize the diffusion model.
        
        Args:
            timesteps: Number of diffusion steps
            beta_schedule: Type of noise schedule ('linear' or 'cosine')
            device: Device to run the model on
        """
        self.timesteps = timesteps
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # Move all tensors to device
        self._move_to_device()
    
    def _move_to_device(self):
        """Move all precomputed tensors to the specified device."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(self.device))
    
    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.028):
        """Linear beta schedule."""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule."""
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0.0001, max=0.9999)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Samples from q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod) * x_0, (1 - alpha_cumprod) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t.to(torch.long)].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t.to(torch.long)].view(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    def p_mean_variance(self, model, x_t, t, y=None, clip_denoised=True):
        """
        Compute mean and variance for the posterior p(x_{t-1} | x_t, y)
        """
        # Ensure t is a float tensor for the model
        t = t.float()
        
        # Ensure y has the same batch size as x_t if provided
        if y is not None and y.shape[0] != x_t.shape[0]:
            # Repeat or slice y to match x_t's batch size
            if y.shape[0] < x_t.shape[0]:
                # Repeat y to match x_t's batch size
                y = y.repeat(x_t.shape[0] // y.shape[0] + 1, 1)[:x_t.shape[0]]
            else:
                # Slice y to match x_t's batch size
                y = y[:x_t.shape[0]]
        
        # Predict noise using the model
        pred_noise = model(x_t, t, y) if y is not None else model(x_t, t)
        
        # Compute the predicted x_0
        pred_x_0 = self._predict_x0_from_noise(x_t, t, pred_noise)
        
        # Clip x_0 for numerical stability
        if clip_denoised:
            pred_x_0 = torch.clamp(pred_x_0, -1., 1.)
        
        # Get the mean for q(x_{t-1} | x_t, x_0)
        pred_mean = self._q_posterior_mean(x_t, pred_x_0, t)
        
        # Get the variance
        posterior_variance_t = self.posterior_variance[t.to(torch.long)].view(-1, 1, 1, 1)
        posterior_log_variance_t = self.posterior_log_variance_clipped[t.to(torch.long)].view(-1, 1, 1, 1)
        
        return pred_mean, posterior_variance_t, posterior_log_variance_t, pred_x_0
    
    def _predict_x0_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and the predicted noise epsilon_theta
        x_0 = (x_t - sqrt(1-alpha_cumprod) * noise) / sqrt(alpha_cumprod)
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t.to(torch.long)].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t.to(torch.long)].view(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def _q_posterior_mean(self, x_t, x_0, t):
        """
        Compute the mean of the posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self.posterior_mean_coef1[t.to(torch.long)].view(-1, 1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t.to(torch.long)].view(-1, 1, 1, 1)
        
        return posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
    
    def p_sample(self, model, x_t, t, y=None, clip_denoised=True):
        """
        Sample from the posterior p(x_{t-1} | x_t, y) for one timestep
        """
        # Compute mean and variance
        pred_mean, variance, log_variance, pred_x0 = self.p_mean_variance(model, x_t, t, y, clip_denoised)
        
        # No noise when t == 0
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        
        # Sample from the posterior
        return pred_mean + torch.exp(0.5 * log_variance) * noise, pred_x0
    
    def p_sample_loop(self, model, shape, y=None, noise=None, show_progress=True):
        """
        Sample from the model using the reverse diffusion process
        """
        # Start from pure noise
        if noise is None:
            x_t = torch.randn(shape, device=self.device)
        else:
            x_t = noise
        
        # Initialize x_0_pred for visualization
        x_0_preds = []
        
        # Create progress bar if requested
        iterator = tqdm(reversed(range(self.timesteps)), total=self.timesteps) if show_progress else reversed(range(self.timesteps))
        
        # Sampling loop
        for t in iterator:
            # Expand t to match batch size
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.int64)
            
            # Sample x_{t-1} from p(x_{t-1} | x_t, y)
            x_t, x_0_pred = self.p_sample(model, x_t, t_batch, y, clip_denoised=True)
            
            # Store x_0 predictions for visualization (optional)
            x_0_preds.append(x_0_pred.detach().cpu())
        
        return x_t, x_0_preds
    
    def train_step(self, model, optimizer, x_0, t, y=None):
        """
        Perform a single training step.
        """
        model.train()
        optimizer.zero_grad()
        
        # Sample noise and compute noisy sample x_t
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        
        # Predict noise
        if y is not None:
            pred_noise = model(x_t, t.float(), y)
        else:
            pred_noise = model(x_t, t.float())
        
        # Compute loss
        loss = nn.MSELoss()(pred_noise, noise)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()


def train_and_sample(
    diffusion_model,
    unet_model,
    train_loader,
    num_epochs=120,
    lr=1e-4,
    save_dir='./results',
    sample_every=5,
    save_model_every=10,
    num_samples=5,
    sample_shape=None,
    sample_classes=None
):
    """
    Train the diffusion model and sample from it periodically.
    """
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    
    # Set up Tensorboard writer
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    
    # Optimizer
    optimizer = optim.Adam(unet_model.parameters(), lr=lr)
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Process batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch, labels in progress_bar:
            x_0 = batch.to(diffusion_model.device)
            y = labels.to(diffusion_model.device).float() if labels is not None else None
            if y is not None:
                y= torch.functional.F.one_hot(y.to(torch.int64), num_classes=2)
                y = y.to(torch.float32)

            # Sample random timesteps
            t = torch.randint(
                0, diffusion_model.timesteps, (x_0.shape[0],), 
                device=diffusion_model.device
            )
            
            # Train step
            loss = diffusion_model.train_step(unet_model, optimizer, x_0, t, y)
            
            # Update stats
            epoch_loss += loss
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss)
            
            # Log to Tensorboard
            writer.add_scalar('train/batch_loss', loss, global_step)
        
        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
        
        # Log to Tensorboard
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        
        # Sample from the model
        if (epoch + 1) % sample_every == 0 or epoch == num_epochs - 1:
            print(f"Generating samples for epoch {epoch+1}...")
            
            with torch.no_grad():
                for class_idx in sample_classes:
                    #
                    # For one-hot encodings
                    y = torch.zeros((num_samples, len(sample_classes)), device=diffusion_model.device)
                    y[:, class_idx] = 1.0
                    print(y.shape)
                    
                    # Generate samples
                    samples, _ = diffusion_model.p_sample_loop(
                        unet_model, 
                        shape=(num_samples, *sample_shape[1:]), 
                        y=y
                    )
                    
                    # Save samples
                    samples_path = os.path.join(save_dir, 'samples', f'epoch_{epoch+1}_class_{class_idx}.pt')
                    torch.save(samples.cpu(), samples_path)
        
        # Save model
        if (epoch + 1) % save_model_every == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(save_dir, 'models', f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, model_path)
            print(f"Model saved to {model_path}")
    
    # Close Tensorboard writer
    writer.close()
    return unet_model


def generate_samples(
    diffusion_model,
    unet_model,
    num_samples_per_class=10,
    sample_shape=None,
    sample_classes=None,
    save_dir='./results/final_samples'
):
    """
    Generate samples for each class after training.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_samples = []
    all_labels = []
    
    for class_idx in sample_classes:
        print(f"Generating {num_samples_per_class} samples for class {class_idx}...")
        
        # Create class label for this specific class
        
        # For one-hot encodings
        y = torch.zeros((num_samples_per_class, len(sample_classes)), device=diffusion_model.device)
        y[:, class_idx] = 1.0
        
        # Generate samples
        with torch.no_grad():
            samples, _ = diffusion_model.p_sample_loop(
                unet_model, 
                shape=(num_samples_per_class, *sample_shape[1:]), 
                y=y
            )
        
        # Save samples
        samples_path = os.path.join(save_dir, f'class_{class_idx}_samples.pt')
        torch.save(samples.cpu(), samples_path)
        
        # Add to collection
        all_samples.append(samples.cpu())
        all_labels.extend([class_idx] * num_samples_per_class)
    
    # Combine all samples and save
    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.tensor(all_labels)
    
    combined_path = os.path.join(save_dir, 'all_samples.pt')
    torch.save({
        'samples': all_samples,
        'labels': all_labels
    }, combined_path)
    
    print(f"All samples saved to {combined_path}")
    return all_samples, all_labels


def main():
    # Set random seed for reproducibility
    torch.manual_seed(99)
    np.random.seed(99)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define parameters
    timesteps = 1000  # Reduced from 2000 for faster training
    batch_size = 64
    num_epochs = 120
    sample_shape = (1, 4, 12, 8)  # From the original code
    sample_classes = [0, 1]  # Two classes from the original code
    
    # Load dataset
    dataset = ONRData()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0][0].shape}")
    
    # Create diffusion model
    diffusion_model = DiffusionModel(timesteps=timesteps, beta_schedule='linear', device=device)
    
    # Create UNet model
    unet_model = UNet(c_in=4, c_out=4, time_dim=64, num_classes=len(sample_classes), device=device).to(device)
    
    # Check your UNet model to determine the expected format for label conditioning
    # Based on the error, let's first train without sampling to avoid issues
    # Then we'll implement proper sampling based on the UNet's requirements
    
    # First train for a few epochs to understand model behavior
    try:
        print("Starting training...")
        # for epoch in range(2):  # Just 2 epochs to test
        #     total_loss = 0.0
        #     num_batches = 0
            
        #     for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/2"):
        #         x_0 = batch.to(device)
        #         y = labels.to(device)
        #         y= torch.functional.F.one_hot(y.to(torch.int64), num_classes=2)
        #         y = y.to(torch.float32)

                
        #         # Sample random timesteps
        #         t = torch.randint(0, timesteps, (x_0.shape[0],), device=device)
                
        #         # Train step
        #         loss = diffusion_model.train_step(unet_model, optim.Adam(unet_model.parameters(), lr=1e-4), x_0, t, y)
                
        #         total_loss += loss
        #         num_batches += 1
            
        #     avg_loss = total_loss / num_batches
        #     print(f"Epoch [{epoch+1}/2], Avg Loss: {avg_loss:.4f}")
        
        print("Initial training successful. Now proceeding with full training...")
        
        # Get a sample input and label to check model's handling
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device).float()
        sample_y= torch.functional.F.one_hot(sample_y.to(torch.int64), num_classes=2)
        sample_y = sample_y.to(torch.float32)

        
        t_sample = torch.zeros(sample_x.shape[0], device=device, dtype=torch.int64)
        
        # Test model forward pass with sample data
        with torch.no_grad():
            # This will help understand the expected input format
            noise_pred = unet_model(sample_x, t_sample.float(), sample_y)
            print(f"Model forward pass successful. Output shape: {noise_pred.shape}")
        
        # Now proceed with full training and sampling
        trained_model = train_and_sample(
            diffusion_model=diffusion_model,
            unet_model=unet_model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            lr=1e-4,
            save_dir='./diffusion_results',
            sample_every=5,
            save_model_every=10,
            num_samples=5,
            sample_shape=sample_shape,
            sample_classes=sample_classes
        )
        
        # Generate final samples
        generate_samples(
            diffusion_model=diffusion_model,
            unet_model=trained_model,
            num_samples_per_class=100,  # 100 samples per class as in the original code
            sample_shape=sample_shape,
            sample_classes=sample_classes,
            save_dir='./diffusion_results/final_samples'
        )
    
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        # Let's attempt to diagnose the UNet model structure
        print("\nAttempting to diagnose UNet structure...")
        
        # Analyze the UNet to determine how it handles labels
        # This is to understand if it expects one-hot encoded labels or single class indices
        if hasattr(unet_model, 'embed_y'):
            print("Model has label embedding layer (embed_y)")
            
            # Get the size of the label embedding
            for name, param in unet_model.named_parameters():
                if 'embed_y' in name:
                    print(f"Label embedding parameters: {name}, shape: {param.shape}")
        
        if hasattr(unet_model, 'forward'):
            print("\nModel's forward method signature:")
            import inspect
            print(inspect.signature(unet_model.forward))
            
        # Print basic model structure
        # print("\nBasic UNet structure:")
        # print(unet_model)
        
        


if __name__ == "__main__":
    main()