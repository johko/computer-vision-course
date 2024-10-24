# ResNet (Residual Network)




Neural networks with more layers were assumed to be more effective because adding more layers improves the model performance.

As the networks became deeper, the extracted features could be further enriched, such as seen with VGG16 and VGG19.

A question arose: "Is learning networks as easy as stacking more layers"?
An obstacle to answering this question, the gradient vanishing problem, was addressed by normalized initializations and intermediate normalization layers.

However, a new issue emerged: the degradation problem. As the neural networks became deeper, accuracy saturated and degraded rapidly. An experiment comparing shallow and deep plain networks revealed that deeper models exhibited higher training and test errors, suggesting a fundamental challenge in training deeper architectures effectively. This degradation was not because of overfitting but because the training error increased when the network became deeper. The added layers did not approximate the identity function.

<img src="https://i.sstatic.net/xDLez.jpg" alt="Degradation-error" width="512" height="512">


ResNet’s residual connections unlocked the potential of the extreme depth, propelling the accuracy upwards compared to the previous architectures.



## ResNet Architecture

- A Residual Block. Source: ResNet Paper
![residual](https://huggingface.co/datasets/hf-vision/course-assets/blob/main/ResnetBlock.png)

ResNets introduce a concept called residual learning, which allows the network to learn the residuals (i.e., the difference between the learned representation and the input), instead of trying to directly map inputs to outputs. This is achieved through skip connections (or shortcut connections). 


Let's break this down:

#### 1. Basic Building Block: Residual Block

In a typical neural network layer, we aim to learn a mapping F(x), where x is the input and F is the transformation the network applies. Without residual learning, the transformation at a layer is: y = F(x). In ResNets, instead of learning F(x) directly, the network is designed to learn the residual R(x), where: R(x) = F(x) − x. Thus, the transformation in a residual block is written as: y = F(x) + x.

Here, x is the input to the residual block, and F(x) is the output of the block's stacked layers (usually convolutions, batch normalization, and ReLU). The identity mapping x is directly added to the output of F(x) (through the skip connection). So the block is learning the residual R(x) = F(x), and the final output is F(x) + x. This residual function is easier to optimize compared to the original mapping. If the optimal transformation is close to the identity (i.e., no transformation is needed), the network can easily set F(x) ≈ 0 to pass through the input as it is.



* The building block of ResNet can be shown in the picture, source ResNet paper. 

![resnet_building_block](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ResnetBlock.png)


#### 2. Learning Residuals and Gradient Flow

The primary reason ResNets work well for very deep networks is due to improved gradient flow during backpropagation. Let’s analyze the backward pass mathematically. Let's say you have the residual block y = F(x) + x. When calculating gradients via the chain rule, we compute the derivative of the loss 𝐿 with respect to the input x. For a standard block (without residuals), the gradient is:

`∂x/∂L = ∂F(x)/∂L * ∂F(x)/∂x`

However, for the residual block, where y = F(x) + x, the gradient becomes:

`∂L/∂x = ∂L/∂F(x) * ∂F(x)/∂x + ∂L/∂F(x) ⋅1`

Notice that we now have an additional term 1 in the gradient calculation. This means that the gradient at each layer has a direct path back to earlier layers, improving gradient flow and reducing the chance of vanishing gradients. The gradients flow more easily through the network, allowing deeper networks to train without degradation.

#### 3. Why Deeper Networks are Now Possible:

ResNets made it feasible to train networks with hundreds or even thousands of layers. Here’s why deeper networks benefit from this:

* Identity Shortcut Connections: Shortcut connections perform identity mapping and their output is added to the output of the stacked layers. Identity shortcut connections add neither extra parameters nor
computational complexity, these connections bypass layers, creating direct paths for information flow, and they enable neural networks to learn the residual function (F).
* Better Gradient Flow: As explained earlier, the residuals help gradients propagate better during backpropagation, addressing the vanishing gradient problem in very deep networks.
* Easier to Optimize: By learning residuals, the network is essentially breaking down the learning process into easier, incremental steps. It’s easier for the network to learn the residual R(x) = F(x) − x than it is to learn F(x) directly, especially in very deep networks.


### Summarizing:

We can conclude that ResNet Network ->  Plain Network + Shortcuts!

For operations  \(F(x) + x\), \(F(x)\) and \(x\) should have identical dimensions.
ResNet employs two techniques to achieve this:

- Zero-padding shortcuts that add channels with zero values, maintaining dimensions without introducing extra parameters to be learned.
- Projection shortcuts that use 1x1 convolutions to adjust dimensions when necessary, involving some additional learnable parameters.

In deeper ResNet architectures like ResNet 50, 101, and 152, a specialized "bottleneck building block" is employed to manage parameter complexity and maintain efficiency while enabling even deeper learning.

## ResNet Code



### Deep Residual Networks Pre-trained on ImageNet
Below you can see how to load pre-trained ResNet with an image classification head using transformers library.
```python
from transformers import ResNetForImageClassification

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

model.eval()
```
All pre-trained models expect input images normalized similarly, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded into a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].


Here's a sample execution. This example is available in the [Hugging Face documentation](https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/resnet).

```python
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```


## References

- [PyTorch docs](https://pytorch.org/hub/pytorch_vision_resnet/)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

- [Resnet Architecture Source: ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Hugging Face Documentation on ResNet](https://huggingface.co/docs/transformers/en/model_doc/resnet)
