# nanovit

## What are Vision Transformers?

A Vision Transformer (ViT) treats an image as a sequence of patch tokens and processes those tokens with transformer blocks.

In a plain ViT:
1. the image is split into fixed-size patches
2. each patch becomes an embedding vector
3. global self-attention mixes information across all patches
4. a classifier head predicts the label

This works very well, but plain ViT models can be expensive because:
- the token sequence can be long
- global attention scales poorly with image size
- early image processing benefits from convolutional inductive biases

## What makes TinyViT-style models different from plain ViT?

TinyViT-style models are designed to be **smaller and more efficient** than a plain ViT while keeping transformer-style reasoning.

This repo demonstrates the main ideas:

- **convolutional patch stem** instead of raw patchify-only embedding
- **MBConv-style early stage** on feature maps
- **hierarchical processing** with spatial downsampling between stages
- **local window attention** instead of always attending globally
- **relative position bias** inside each attention window
- **depthwise local convolution** between attention and the MLP
- **global mean pooling** instead of a class token