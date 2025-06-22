import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

# --- Configuration (MUST match training script) ---
LATENT_DIM = 100
EMBED_DIM = 10
NUM_CLASSES = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
MODEL_PATH = 'generator_cgan.pth' # Path to your trained model file

# --- Model Architecture (MUST match the Generator in training script) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, img_size, num_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_channels = num_channels

        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True)
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # Output: 128 x 14 x 14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False), # Output: num_channels x 28 x 28
            nn.Tanh() # Outputs pixel values in [-1, 1]
        )

    def forward(self, noise, labels):
        gen_input_label = self.label_embedding(labels)
        gen_input = torch.cat((noise, gen_input_label), 1) # Concat along feature dimension
        
        gen_input = self.fc(gen_input)
        gen_input = gen_input.view(-1, 256, 7, 7) # Reshape to 256 channels, 7x7 spatial
        
        img = self.deconv_layers(gen_input)
        return img

# --- Load the trained Generator model ---
@st.cache_resource # Cache the model loading to avoid re-loading on every rerun
def load_generator_model(model_path):
    generator = Generator(LATENT_DIM, NUM_CLASSES, EMBED_DIM, IMAGE_SIZE, NUM_CHANNELS)
    # Load model state dict, mapping to CPU as Streamlit Cloud might not have GPU
    try:
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        generator.eval() # Set to evaluation mode
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory as app.py.")
        st.stop() # Stop the app execution if model not found
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return generator

# --- Initialize Generator ---
generator = load_generator_model(MODEL_PATH)

# --- Streamlit UI ---
st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")
st.title("🔢 Handwritten Digit Generator")
st.markdown("Generate 5 unique handwritten images for a selected digit (0-9) using a Conditional GAN trained on MNIST.")

# --- User Input ---
selected_digit = st.selectbox(
    "Select a digit to generate:",
    options=list(range(NUM_CLASSES)),
    index=0 # Default to digit 0
)

# --- Generate Button ---
if st.button(f"Generate 5 images of digit {selected_digit}"):
    st.subheader(f"Generated Images for Digit {selected_digit}:")
    
    # Generate 5 images
    generated_images = []
    with torch.no_grad(): # No need to calculate gradients during inference
        for _ in range(5): # Generate 5 images
            noise = torch.randn(1, LATENT_DIM) # Generate one noise vector
            # Create a label tensor for the selected digit
            labels = torch.LongTensor([selected_digit]) 

            # Generate image (ensure noise and labels are on CPU for inference if model loaded to CPU)
            generated_img_tensor = generator(noise, labels).cpu().squeeze(0) # Remove batch dim and channel if 1

            # Denormalize from [-1, 1] to [0, 1] and then to [0, 255] for PIL
            generated_img_tensor = (generated_img_tensor + 1) / 2.0
            generated_img_array = (generated_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8) # HWC format for PIL

            # If it's a grayscale image (1 channel), remove the channel dimension
            if generated_img_array.shape[2] == 1:
                generated_img_array = generated_img_array.squeeze(2)

            img_pil = Image.fromarray(generated_img_array, 'L') # 'L' for grayscale

            generated_images.append(img_pil)
    
    # Display images in columns
    cols = st.columns(5) # Create 5 columns
    for i, img in enumerate(generated_images):
        with cols[i]:
            st.image(img, caption=f"Digit {selected_digit} ({i+1})", use_column_width=True)

st.markdown("---")
st.markdown("Model trained on MNIST dataset using PyTorch. Deployed with Streamlit.")