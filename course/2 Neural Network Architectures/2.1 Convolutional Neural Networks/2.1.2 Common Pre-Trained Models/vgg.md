
# Very Deep Convolutional Networks for Large Scale Image Recognition (2014)


1. Introduction

The VGG architecture was developed in 2014 by Karen Simonyan and Andrew Zisserman from the Visual Geometry Group at Oxford University. The authors detailed key advancements in convolutional neural networks for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). VGG demonstrated significant improvements with increased depth and repeated small (3X3) convolutional filters. It was a research project to systematically evaluate prior art configurations and identify key structural designs. 



2. VGG Network Architechture 
   
- Input is 224X224 image. 
- 3X3 convolution kernels with 2X2 max pooling layers.
- Number of channels for each convolutional layer 64 -> 128 -> 256 -> 512 -> 512. 
- VGG16 has 16 hidden layers (13 convolutional layers and 3 fully connected layers).
- VGG19 has 19 hidden layers (16 convolutional layers and 3 fully connected layers).

Below is the network structure of VGG16, The activation and dropout layers were omitted for brevity. MaxPooling layers were included to display changes in output tensor size at each layer. 

|Layer Type |Filter Size|Output Size|No. Params|
|-----------|------------|----------|----------|
|Input Image|0|3X224X224|0|
|Conv2D|3X3|64X224X224|1,792|
|Conv2D|3X3|64X224X224|39,628|
|MaxPool2D|2X2|64X112X112|0|
|Conv2D|3X3|128X112X112|73,856|
|Conv2D|3X3|128X112X112|147,584|
|MaxPool2D|2X2|128X56X56|0|
|Conv2D|3X3|256X56X56|295,168|
|Conv2D|3X3|256X56X56|590,080|
|Conv2D|3X3|256X56X56|590,080|
|MaxPool2D|2X2|256X28X28|0|
|Conv2D|3X3|512X28X28|1,180,160|
|Conv2D|3X3|512X28X28|2,359,808|
|Conv2D|3X3|512X28X28|2,359,808|
|MaxPool2D|2X2|512X14X14|0|
|Conv2D|3X3|512X14X14|2,359,808|
|Conv2D|3X3|512X14X14|2,359,808|
|Conv2D|3X3|512X14X14|2,359,808|
|MaxPool2D|2X2|512X7X7|0|
|Linear|N/A|4096|102,764,544|
|Linear|N/A|4096|16,781,312|
|Linear|N/A|1000|4,096,000|



3. Key Comparisons 

- VGG (16 or 19 layers) is relatively deeper than other SOTA networks at the time. AlexNet, winning model for ILSVRC 2012 only has 8 layers.    
- Multiple small (3X3) receptive field filters with ReLU activation instead of one large (7X7 or 11X11) filter lead to better learning of complex features. Smaller filters also mean fewer parameters per layer, with additional nonlinearity introduced in between.   
- Multiscale training and inference. Each image was trained in multiple rounds with varying scales to ensure similar characteristics were captured at different sizes.  
- Consistency and simplicity of the VGG network make it easier to scale or modify for future improvements. 


4. Pytorch Example 

```

```

 