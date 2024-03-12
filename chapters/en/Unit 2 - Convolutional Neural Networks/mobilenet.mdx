# MobileNet

MobileNet is a type of neural network architecture designed for mobile devices. It was developed by Google's research team and first introduced in 2017. The primary goal of MobileNet is to provide high-performance, low-latency image classification and object detection on smartphones, tablets, and other resource-constrained devices.  
  
MobileNet achieves this by using depthwise separable convolutions, which are a more efficient alternative to standard convolutions. Depthwise separable convolutions break down the computation into two separate steps: depthwise convolution followed by pointwise convolution. This results in a significant reduction of parameters and computational complexity, allowing MobileNet to run efficiently on mobile devices.  

## Convolution Types on MobileNet

By replacing regular convolutional layers with these depthwise separable convolutions and pointwise convolutions, MobileNet achieves high accuracy while minimizing computational overhead, making it well-suited for mobile devices and other resource-limited platforms. There are two types of convolutions used in MobileNet:

### Depthwise Seperable Convolutions

In traditional convolutional layers, each filter applies its weight across all input channels simultaneously. Depthwise separable convolutions break this down into two steps: depthwise convolution followed by pointwise convolution.  

This step performs a convolution separately for each channel (a single color or feature) in the input image using a small filter (usually 3x3). The output of this step is the same size as the input but with fewer channels.

### Pointwise Seperable Convolutions

This type of convolution applies a single filter (usually 1x1) across all channels in both input and output layers. It has fewer parameters than regular convolution and can be seen as an alternative to fully connected layers, making it suitable for mobile devices that have limited computational resources.

After depthwise convolution, this step combines the filtered outputs from previous steps using another 1x1 convolutional layer. This operation effectively aggregates the features learned by the depthwise convolutions into a smaller set of features, reducing the overall complexity while retaining important information.

### Why Do We Use These Convolutions Instead of Regular Convolutions?
To better understand, let's simplify it and explain:

#### Regular Convolutions, a big, All-in-one Filter

Imagine you have a big, thick filter (like a sponge with many layers). This filter is applied over the entire image. It processes all parts of the image and all its features (like colors) at once. This requires a lot of work (computation) and a big filter (memory).

#### Depthwise Separable Convolutions - Two-Step, Lighter Process:
MobileNet does this process basic. It splits the big filter into two smaller, simpler steps:

 - **Step 1 - Depthwise Convolution:** First, it uses a thin filter (like a single layer of a sponge) for each feature of the image
   separately (like processing each color individually). This is less work because each filter is small and simple.
   
 - **Step 2 - Pointwise Convolution:** Then, it uses another small filter (just a tiny dot) to mix these features back together. This step is like taking a summary of what the first filters found.

#### What are these steps about?

MobileNet uses these two smaller steps instead of one big step, it's like doing a lighter version of the work needed in regular convolutions. It's more efficient, especially on devices that aren't very powerful, like smartphones.

With smaller filters, MobileNet doesn't need as much memory. It's like needing a smaller box to store all your tools, making it easier to fit into smaller devices.

### How Do 1x1 Convolutions Work Compared to Normal Convolutions?

#### Normal Convolutions

-   Normal convolutions use a larger kernel (like 3x3 or 5x5) to look at a group of pixels in the image at once. It's like observing a small patch of a picture to understand a part of the scene.
-   These convolutions are good at understanding features like edges, corners, and textures by analyzing how pixels are arranged next to each other.

#### 1x1 Convolutions

-  A 1x1 convolution, looks at one pixel at a time. It doesn't try to understand the arrangement of neighboring pixels.
-  Even though it's looking at one pixel, it considers all the information from different channels (like the RGB channels in a color image). It combines these channels to create a new version of that pixel.
-  The 1x1 convolution can either increase or decrease the number of channels. This means it can simplify the information (by reducing channels) or create more complex information (by increasing channels).

#### Key Differences

-   **Area of Focus:** Normal convolutions analyze a set of pixels together to understand patterns, whereas 1x1 convolutions focus on individual pixels, combining the information from different channels.
-   **Purpose:** Normal convolutions are used for detecting patterns and features in an image. In contrast, 1x1 convolutions are mainly used to alter the channel depth, helping in adjusting the complexity of the information for subsequent layers in a neural network for efficiency in weak devices.
 
MobileNet also employs techniques like channel-wise linear bottleneck layers, which improve model accuracy while reducing the number of parameters. The architecture is designed with optimizations for various hardware platforms, including CPUs, GPUs, and even specialized hardware such as Google's Tensor Processing Units (TPUs).

### Channel-wise Linear Bottleneck Layers
Channel-wise linear bottleneck layers help to further reduce the number of parameters and computational cost while maintaining high accuracy in image classification tasks.

A channel-wise linear bottleneck layer consists of three main operations applied sequentially:  
1. **Depthwise convolution:** This step performs a convolution separately for each channel (a single color or feature) in the input image using a small filter (usually 3x3). The output of this step is the same size as the input but with fewer channels.  
2. **Batch normalization:** This operation normalizes the activation values across each channel, helping to stabilize the training process and improve generalization performance.  
3. **Activation function:** Typically, a ReLU (Rectified Linear Unit) activation function is used after batch normalization to introduce non-linearity in the network.

### What does ReLU do?

Some problems may occur during training. We will explain them first, then we explain what ReLU does to this problems. 

#### Vanishing Gradient Problem

In deep neural networks, especially during backpropagation, the vanishing gradient problem can occur. This happens when gradients (which are used to update the network's weights) become very small as they are passed back through the network's layers. If these gradients become too small, they "vanish," making it hard for the network to learn and adjust its weights effectively.

ReLU has a linear, non-saturating form for positive values (where it simply outputs the input if it's positive), it ensures that the gradients do not become too small, allowing for faster learning and more effective weight adjustments in the network.

#### Non-Linearity

Without non-linearity, a neural network, no matter how many layers it has, would function as a linear model, incapable of learning complex patterns. 

Non-linear functions like ReLU enable neural networks to capture and learn complex relationships in the data.

### Inference

You can use Hugging Face transformers to infer with a variation of transformers models like below:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# initialize processor and model
preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

# preprocess the inputs
inputs = preprocessor(images=image, return_tensors="pt")

# get the output and the class labels
outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

```
Predicted class: tabby, tabby cat
```

### Example Implementation using PyTorch

You can find an example implementation of MobileNet for PyTorch below:

```python  
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        # MobileNet body
        self.dw_conv2 = DepthwiseSeparableConv(32, 64, 1)
        self.dw_conv3 = DepthwiseSeparableConv(64, 128, 2)
        self.dw_conv4 = DepthwiseSeparableConv(128, 128, 1)
        self.dw_conv5 = DepthwiseSeparableConv(128, 256, 2)
        self.dw_conv6 = DepthwiseSeparableConv(256, 256, 1)
        self.dw_conv7 = DepthwiseSeparableConv(256, 512, 2)

        # 5 depthwise separable convolutions with stride 1
        self.dw_conv8 = DepthwiseSeparableConv(512, 512, 1)
        self.dw_conv9 = DepthwiseSeparableConv(512, 512, 1)
        self.dw_conv10 = DepthwiseSeparableConv(512, 512, 1)
        self.dw_conv11 = DepthwiseSeparableConv(512, 512, 1)
        self.dw_conv12 = DepthwiseSeparableConv(512, 512, 1)

        self.dw_conv13 = DepthwiseSeparableConv(512, 1024, 2)
        self.dw_conv14 = DepthwiseSeparableConv(1024, 1024, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.dw_conv2(x)
        x = F.relu(x)
        x = self.dw_conv3(x)
        x = F.relu(x)
        x = self.dw_conv4(x)
        x = F.relu(x)
        x = self.dw_conv5(x)
        x = F.relu(x)
        x = self.dw_conv6(x)
        x = F.relu(x)
        x = self.dw_conv7(x)
        x = F.relu(x)

        x = self.dw_conv8(x)
        x = F.relu(x)
        x = self.dw_conv9(x)
        x = F.relu(x)
        x = self.dw_conv10(x)
        x = F.relu(x)
        x = self.dw_conv11(x)
        x = F.relu(x)
        x = self.dw_conv12(x)
        x = F.relu(x)

        x = self.dw_conv13(x)
        x = F.relu(x)
        x = self.dw_conv14(x)
        x = F.relu(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Create the model
mobilenet = MobileNet(num_classes=1000)
print(mobilenet)
```

You can also find a pretrained MobileNet model checkpoint on this HuggingFace [link](https://huggingface.co/google/mobilenet_v2_1.0_224).