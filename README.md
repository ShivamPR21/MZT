# ModuleZooTorch

```diff
- Note: This library is deprecated for a faster and more scalable alternative (i.e. MZX),
- there will be no further changes in this library.
```

> The `MZX` library will be released on the following link [MZX](https://github.com/Matroid-ASI/MZX.git)

## MZT Details

Collection of neural-net modules including both general blocks, and those introduced in latest research

> The library is still under development, if you have some cool NN blocks in mind that deserves to be implemented in MZT please don't hesitate to open an issue with feature request, I'll try my best to implement those ASAP.

## Installation

```shell
pip install mzt
```

## Available Modules

- Convolution Blocks:
  - ConvNormActivation1d
  - ConvNormActivation2d
  - ConvInvertedBlock1d
  - ConvInvertedBlock2d
- Residual Blocks:
  - ConvResidualBlock1d
  - ConvResidualBlock2d
  - ConvBottleNeckResidualBlock1d
  - ConvBottleNeckResidualBlock2d
  - ConvInvertedResidualBlock1d
  - ConvInvertedResidualBlock2d
- Attention Blocks:
  - SelfAttention1d
  - SelfAttention2d
  - MultiHeadSelfAttention1d
  - MultiHeadSelfAttention2d
  - MultiHeadAttention1d
  - MultiHeadAttention2d
  - MultiHeadAttentionLinear
- Graphical/Geometric Blocks:
  - GraphConv2d
