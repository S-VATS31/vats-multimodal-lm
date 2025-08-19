# Vats Multi-Modal Language Model
This repository contains complete code to a state-of-the-art multi-modal large language model supporting text generation, image processing + generation, video processing + generation, audio processing + generation, and more to come.

# Table of Contents
Table of contents goes here

# Introduction
Here I present Vats Multimodal Language Model (LM), a multi-modal language model used to support all modalities mentioned above. 
- Cost-Effective Token Generation: I present an Mixture of Experts (MoE) language model using Flash Attention V2, Grouped Query Attention (GQA), Sliding Window Attention (SWA), Key Value Caching (KV Caching) and more features to support an advanced language model while keeping inference costs efficient. 
- 2D Image Processing: For image processing, I use an encoder-only 2D Vision Transformer (ViT) utilizing spatial attention where we apply Flash Attention V2, GQA, and SWA over the flattened height and width dimensions. We use a Conv2D layer to create image patches.
- 3D Video Processing: For video processing, I use an encoder-only 3D ViT utilizing factorized attention. A variant of classic spatio-temporal attention where we apply attention spatially as 1 x H x W and temporal attention as T x 1 x 1. We do this to avoid quadratic complexity all dimensions. Once again, we use a convolutional layer, a Conv3D layer to be exact, to create video patches containing pT frames, pH pixels, and pW pixels.
- 2D Image Generation: We 