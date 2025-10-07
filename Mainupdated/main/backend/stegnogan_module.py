"""
StegnoGAN++ Module
Advanced steganography using Generative Adversarial Networks
Based on deep learning principles for improved steganography
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import logging
from typing import List
import glob

# --- NEW: Check for torchvision dependency ---
try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: The 'torchvision' library is not installed. Data augmentation for training is disabled. Please install it using: pip install torchvision")
# --- END NEW ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StegnoGANGenerator(nn.Module):
    """
    Generator network for StegnoGAN++
    Creates stego images from cover images and secret messages
    """
    
    def __init__(self, input_channels=3, output_channels=3, hidden_dim=64):
        super(StegnoGANGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Message embedding layer
        self.message_embedding = nn.Sequential(
            nn.Linear(1024, hidden_dim * 8 * 8 * 8), # Adjusted for 128x128 image -> 8x8 bottleneck
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Start from 8x8 bottleneck
            nn.ConvTranspose2d(hidden_dim * 8 * 2, hidden_dim * 8, 4, 2, 1), # Concatenated input
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim, output_channels, 4, 2, 1),
            nn.Tanh() # Outputs in [-1, 1] range
        )
        
    def forward(self, cover_image, secret_message):
        # Encode cover image
        encoded = self.encoder(cover_image)
        
        # Embed secret message
        message_embedded = self.message_embedding(secret_message)
        message_embedded = message_embedded.view(encoded.size(0), -1, 8, 8)
        
        # Combine encoded image with embedded message by concatenating
        combined = torch.cat([encoded, message_embedded], dim=1)
        
        # Decode to stego image
        stego_image = self.decoder(combined)
        
        return stego_image

class StegnoGANDiscriminator(nn.Module):
    """
    Discriminator network for StegnoGAN++
    Distinguishes between cover and stego images
    """
    
    def __init__(self, input_channels=3, hidden_dim=64):
        super(StegnoGANDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            # Input is 128 x 128
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1), # 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1), # 32 x 32
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1), # 16 x 16
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1), # 8 x 8
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, 4, 2, 1), # 4 x 4
            nn.BatchNorm2d(hidden_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 16, 1, 4, 1, 0), # 1 x 1
            nn.Sigmoid()
        )
        
    def forward(self, image):
        return self.discriminator(image)

class StegnoGANExtractor(nn.Module):
    """
    Extractor network for StegnoGAN++
    Extracts secret messages from stego images
    """
    
    def __init__(self, input_channels=3, hidden_dim=64, message_length=1024):
        super(StegnoGANExtractor, self).__init__()
        
        self.extractor = nn.Sequential(
            # Input is 128 x 128
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1), # 64 x 64
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1), # 32 x 32
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1), # 16 x 16
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1), # 8 x 8
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, hidden_dim * 16, 4, 2, 1), # 4 x 4
            nn.BatchNorm2d(hidden_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 16, message_length),
            nn.Sigmoid()
        )
        
    def forward(self, stego_image):
        return self.extractor(stego_image)

class ImageAugmentationDataset(Dataset):
    """Custom PyTorch Dataset for loading and augmenting images on the fly."""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}. Skipping.")
                return None
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.warning(f"Skipping corrupt or unreadable image {image_path}: {e}")
            return None

class StegnoGANHandler:
    """
    Main handler for StegnoGAN++ operations
    Manages training, embedding, and extraction
    """
    
    def __init__(self, device='cpu', image_size=128, msg_len=1024):
        self.device = device
        self.image_size = image_size
        self.msg_len = msg_len
        self.generator = StegnoGANGenerator().to(device)
        self.discriminator = StegnoGANDiscriminator().to(device)
        self.extractor = StegnoGANExtractor(message_length=msg_len).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.e_optimizer = optim.Adam(list(self.generator.parameters()) + list(self.extractor.parameters()), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Training state
        self.is_trained = False
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for StegnoGAN++"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            
            image_array = (np.array(image) / 127.5) - 1.0
            image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
            
            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess tensor back to image"""
        try:
            image_array = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            
            image_array = (image_array + 1) * 127.5
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            return image_array
        except Exception as e:
            logger.error(f"Error postprocessing image: {e}")
            return None
    
    def encode_message(self, message: str) -> torch.Tensor:
        """Encode message to a binary tensor"""
        try:
            message_bytes = message.encode('utf-8')
            message_bits = ''.join(format(byte, '08b') for byte in message_bytes)
            
            if len(message_bits) > self.msg_len:
                raise ValueError(f"Message is too long. Max length is {self.msg_len // 8} bytes.")
            
            message_bits = message_bits.ljust(self.msg_len, '0')
            
            message_tensor = torch.FloatTensor([int(bit) for bit in message_bits]).unsqueeze(0)
            return message_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error encoding message: {e}")
            return None
    
    def decode_message(self, message_tensor: torch.Tensor) -> str:
        """Decode binary tensor back to a string message"""
        try:
            message_bits = ''.join(['1' if x > 0.5 else '0' for x in message_tensor.squeeze().cpu().numpy()])
            
            message_bytes = bytearray()
            for i in range(0, len(message_bits), 8):
                byte_bits = message_bits[i:i+8]
                if len(byte_bits) == 8:
                    message_bytes.append(int(byte_bits, 2))
            
            message = message_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
            return message
        except Exception as e:
            logger.error(f"Error decoding message: {e}")
            return ""
    
    def train(self, cover_images: List[str], epochs: int = 100, batch_size: int = 4):
        """Train StegnoGAN++ on a list of cover image paths with data augmentation."""
        if not TORCHVISION_AVAILABLE:
            logger.error("The 'torchvision' library is required for training. Please install it.")
            return False

        self.generator.train()
        self.discriminator.train()
        self.extractor.train()

        try:
            logger.info(f"Starting StegnoGAN++ training for {epochs} epochs.")
            logger.info(f"Found {len(cover_images)} images for training.")
            
            augmentation_transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            train_dataset = ImageAugmentationDataset(cover_images, transform=augmentation_transforms)
            
            def collate_fn(batch):
                batch = list(filter(lambda x: x is not None, batch))
                if not batch:
                    return None
                return torch.utils.data.dataloader.default_collate(batch)

            if len(train_dataset) == 0:
                logger.error("No valid training images could be loaded.")
                return False

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            logger.info(f"Successfully created a dataset loader. Starting training loops...")
            
            last_d_loss, last_g_e_loss = None, None

            for epoch in range(epochs):
                for batch_idx, cover_batch in enumerate(train_loader):
                    try:
                        if cover_batch is None or len(cover_batch) == 0:
                            logger.warning(f"Skipping empty or corrupt batch at epoch {epoch}, index {batch_idx}.")
                            continue
                        
                        b_size = cover_batch.size(0)
                        cover_batch = cover_batch.to(self.device)
                        
                        secret_messages = torch.randint(0, 2, (b_size, self.msg_len)).float().to(self.device)
                        
                        # Train Discriminator
                        self.d_optimizer.zero_grad()
                        real_output = self.discriminator(cover_batch)
                        real_labels = torch.ones_like(real_output).to(self.device)
                        d_real_loss = self.bce_loss(real_output, real_labels)
                        
                        fake_images = self.generator(cover_batch, secret_messages)
                        fake_output = self.discriminator(fake_images.detach())
                        fake_labels = torch.zeros_like(fake_output).to(self.device)
                        d_fake_loss = self.bce_loss(fake_output, fake_labels)
                        
                        d_loss = d_real_loss + d_fake_loss
                        d_loss.backward()
                        self.d_optimizer.step()
                        
                        # Train Generator and Extractor
                        self.e_optimizer.zero_grad()
                        stego_images = self.generator(cover_batch, secret_messages)
                        g_output_on_stego = self.discriminator(stego_images)
                        g_loss_adversarial = self.bce_loss(g_output_on_stego, real_labels)
                        g_loss_recon = self.mse_loss(stego_images, cover_batch)
                        extracted_messages = self.extractor(stego_images)
                        e_loss = self.mse_loss(extracted_messages, secret_messages)
                        
                        total_g_e_loss = g_loss_adversarial * 0.001 + g_loss_recon * 0.5 + e_loss
                        total_g_e_loss.backward()
                        self.e_optimizer.step()

                        last_d_loss, last_g_e_loss = d_loss.item(), total_g_e_loss.item()

                    except Exception as batch_error:
                        logger.error(f"Error processing batch {batch_idx} in epoch {epoch}: {batch_error}", exc_info=True)
                        continue
                
                if epoch % 10 == 0:
                    if last_d_loss is not None:
                        logger.info(
                            f"[Epoch {epoch}/{epochs}] "
                            f"Last Batch D_loss: {last_d_loss:.4f} | "
                            f"Last Batch G_E_loss: {last_g_e_loss:.4f} "
                        )
                    else:
                        logger.warning(f"[Epoch {epoch}/{epochs}] No successful batches were processed in this epoch.")
            
            self.is_trained = True
            logger.info("StegnoGAN++ training completed.")
            return True
            
        except Exception as e:
            logger.error(f"A critical error occurred during the training setup: {e}", exc_info=True)
            return False
    
    def embed(self, cover_path: str, secret_message: str, output_path: str) -> bool:
        """Embed a secret message into a cover image."""
        if not self.is_trained:
            logger.warning("Model not trained. Embedding may yield poor results.")
        
        self.generator.eval()
        try:
            cover_tensor = self.preprocess_image(cover_path)
            if cover_tensor is None: return False
            
            message_tensor = self.encode_message(secret_message)
            if message_tensor is None: return False
            
            with torch.no_grad():
                stego_tensor = self.generator(cover_tensor, message_tensor)
            
            stego_array = self.postprocess_image(stego_tensor)
            if stego_array is None: return False
            
            stego_image = Image.fromarray(stego_array)
            stego_image.save(output_path)
            
            logger.info(f"Message successfully embedded and saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error during embedding: {e}", exc_info=True)
            return False
    
    def extract(self, stego_path: str) -> str:
        """Extract a secret message from a stego image."""
        if not self.is_trained:
            logger.warning("Model not trained. Extraction may fail or be inaccurate.")

        self.extractor.eval()
        try:
            stego_tensor = self.preprocess_image(stego_path)
            if stego_tensor is None: return ""
            
            with torch.no_grad():
                extracted_tensor = self.extractor(stego_tensor)
            
            message = self.decode_message(extracted_tensor)
            logger.info("Message extracted successfully.")
            return message
        except Exception as e:
            logger.error(f"Error during extraction: {e}", exc_info=True)
            return ""

    def save_model(self, model_dir: str):
        """Save trained models to a directory."""
        try:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.generator.state_dict(), os.path.join(model_dir, 'generator.pth'))
            torch.save(self.discriminator.state_dict(), os.path.join(model_dir, 'discriminator.pth'))
            torch.save(self.extractor.state_dict(), os.path.join(model_dir, 'extractor.pth'))
            with open(os.path.join(model_dir, 'status.txt'), 'w') as f:
                f.write(str(self.is_trained))
            logger.info(f"Models saved to {model_dir}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_model(self, model_dir: str):
        """Load trained models from a directory."""
        try:
            if not os.path.isdir(model_dir):
                logger.warning(f"Model directory {model_dir} not found.")
                return

            self.generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth'), map_location=self.device))
            self.discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth'), map_location=self.device))
            self.extractor.load_state_dict(torch.load(os.path.join(model_dir, 'extractor.pth'), map_location=self.device))
            
            with open(os.path.join(model_dir, 'status.txt'), 'r') as f:
                self.is_trained = (f.read().strip().lower() == 'true')
            
            logger.info(f"Models loaded from {model_dir}. Trained status: {self.is_trained}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Create a global instance that can be imported
device = 'cuda' if torch.cuda.is_available() else 'cpu'
stegnogan_handler = StegnoGANHandler(device=device)


# --- MODIFICATION FOR DIRECT TRAINING ---
# This block will only run when the script is executed directly from the terminal
if __name__ == '__main__':
    
    # --- STEP 1: CONFIGURE YOUR TRAINING ---
    # <<< EDIT THIS LINE >>>
    # Point this to the folder containing your training images (PNG, JPG, etc.)
    DATASET_FOLDER = "path/to/your/image_folder" 
    
    # Adjust training parameters if needed
    EPOCHS = 40 # Recommended: 30-50 for small datasets
    BATCH_SIZE = 4 # Adjust based on your GPU memory, 2 or 4 is a safe start
    
    # --- STEP 2: THE SCRIPT WILL HANDLE THE REST ---
    
    print("--- Starting StegnoGAN++ Direct Training ---")
    
    if not os.path.isdir(DATASET_FOLDER):
        print(f"\n[ERROR] The specified dataset folder does not exist: {DATASET_FOLDER}")
        print("Please create the folder and fill it with images, or correct the path in the script.")
    else:
        # Find all valid image files in the specified folder
        valid_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(DATASET_FOLDER, ext)))
        
        if len(image_files) == 0:
            print(f"\n[ERROR] No images found in the specified folder: {DATASET_FOLDER}")
            print("Please make sure your images have a valid extension (.png, .jpg, .jpeg, .bmp).")
        else:
            print(f"Found {len(image_files)} images to train on.")
            print(f"Starting training for {EPOCHS} epochs...")
            
            # Start the training process
            stegnogan_handler.train(image_files, epochs=EPOCHS, batch_size=BATCH_SIZE)
            
            # Save the final model
            if stegnogan_handler.is_trained:
                print("Training complete. Saving the model...")
                stegnogan_handler.save_model('stegnogan_model')
                print("Model saved successfully in the 'stegnogan_model' directory.")
            else:
                print("\n[ERROR] Training did not complete successfully. The model was not saved.")

