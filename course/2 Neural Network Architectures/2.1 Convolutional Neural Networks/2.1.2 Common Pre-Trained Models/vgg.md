
# Very Deep Convolutional Networks for Large Scale Image Recognition (2014)


1. Introduction

VGG architecture was developed in 2014 by Karen Simonyan and Andrew Zisserman from Visual Geometry Group at Oxford University. The authors detailed key advancements in convolutional neural networks for ImageNet Large Scale Visual Recognition Challenge (ILSVRC). VGG demonstrated significant improvements with increased depth and repeated small (3X3) convolutional filters. It was a research project to systematically evaluate prior-art configurations and identify key structural designs. 



2. VGG Network Architechture 
   
- Input is 224X224 image. 
- 3X3 convolution kernels with 2X2 max pooling layers.
- Maxpooling with stride = 1 and padding = 0.
- Number of channels for each convolutional layer 64 -> 128 -> 256 -> 512 -> 512. 
- VGG16 has 16 hidden layers (13 convolutional layers and 3 fully connected layers).
- VGG19 has 19 hidden layers (16 convolutional layers and 3 fully connected layers).



3. Key Comparisons 

- VGG (16 or 19 layers) is relatively deeper than other SOTA networks at the time. AlexNet, winning model for ILSVRC 2012 only has 8 layers.    
- Multiple small (3X3) receptive field filters with ReLU activation instead of one large (7X7 or 11X11) filter lead to better learning of complex features. Smaller filters also means fewer parameters per layer, with additional nonlinearity introduced in between.   
- Multiscale training and inference. Each image was trained multiple rounds with varying scales to ensure similar characteristics are captured at different sizes.  
- Consistency and simplicity of VGG network makes it easier to scale or modify for future improvements. 


4. Pytorch Example 

```

```

 