# Vats Multi-Modal Language Model
This repository contains complete code to a state-of-the-art multi-modal large language model supporting text generation, image processing + generation, video processing + generation, audio processing + generation, and more to come.

# Table of Contents
Table of contents goes here

# 1. Introduction
Here we present Vats Multimodal Language Model (LM), a multi-modal language model used to support all modalities mentioned above. 
- **Cost-Effective Token Generation:** we present an Mixture of Experts (MoE) language model using Flash Attention V2, Grouped Query Attention (GQA), Sliding Window Attention (SWA), Key Value Caching (KV Caching) and more features to support an advanced language model while keeping inference costs efficient. 
- **2D Image Processing:** For image processing, we use an encoder-only 2D Vision Transformer (ViT) utilizing spatial attention where we apply Flash Attention V2, GQA, and SWA over the flattened height and width dimensions. We use a Conv2D layer to create image patches.
- **3D Video Processing:** For video processing, we use an encoder-only 3D ViT utilizing factorized attention. A variant of classic spatio-temporal attention where we apply attention spatially as 1 x H x W and temporal attention as T x 1 x 1. We do this to avoid quadratic complexity all dimensions. Once again, we use a convolutional layer, a Conv3D layer to be exact, to create video patches containing pT frames, pH pixels, and pW pixels.
- **Audio Processing:**
- **2D Image Generation:** For generating 2D images, we first take in the input prompt and essentialy 'enrich' it using the LLM from earlier. This allows the prompt that will be encoded to be more feature-rich as well as less ambigious. The enriched prompt is then passed to the text encoder, which will be given to the latent diffusion model. We can finally then decode the output using a Convolutional Neural Network (CNN) via convolutions and residual blocks.
- **3D Video Generation:**
- **Audio Generation:**

# 2. Deep Dive Into Model Architecture
---
### Mixture of Experts Language Model
---
### 2D Image Vision Encoder
---
### 3D Video Vision Encoder
---
### Audio Encoder
---
### 2D Image Diffusion Transformer
---
### 3D Video Diffusion Transformer
---
### Audio Generation

# 3.

# 4.

# 5. Installation
To install the required packages, run the following:
```bash
# Clone the repository
git clone https://github.com/S-VATS31/vats-multimodal-lm.git
cd vats-multimodal-lm

# Install requiremnts
pip install -r requirements.txt
```

# 6. Testing
Since the model is currently in beta, functionality is still being worked out. We will demonstrate model functionality with the model level forward passes of the LLM as well as the 3D Video Encoder. Run the following to ensure the LLM and Vision Encoder both have working forward passes:
```bash
# LLM Forward
# LLM has not been trained, expect nonsense output
python3 src/transformers/nlp/inference/generate.py
# Vision Encoder Forward
python3 src/transformers/vision/vit_3d/model.py
```

# 7. License
This repository is licensed under the MIT license.

# 8. Contact Us
For any questions or issues please fill out the following form: <https://forms.gle/mja1f9SdYLQFLYe26>.