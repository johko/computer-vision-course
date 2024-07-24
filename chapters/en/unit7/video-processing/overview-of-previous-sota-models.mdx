# Overview of the previous SOTA models
### General Trend:

The success of Deep Learning, particularly CNNs trained on massive datasets like ImageNet, revolutionized image recognition. This trend continues in video processing. However, video data introduces another dimension compared to static images: time. This simple change introduced a new set of challenges that CNN trained in static images were not built to deal with. 

# Previous SOTA Models in Video Processing

## Two-Stream Network(2014):

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/previous sota models/SOTA Models Two-Stream architecture for video classification.png" alt="Two-Stream architecture for video classification">
</div>

This paper extended Deep Convolutional Networks(ConvNets) to perform action-recognition in video data. 

The proposed architecture is called Two-Stream Network. It uses two separate pathways within a neural network:

- **Spatial Stream:** A standard 2D CNN processes individual frames to capture appearance information.
- **Temporal Stream:** A 2D CNN, or another network, that processes several frame sequences (optical flow) to capture motion information.
- **Fusion:** The outputs from both streams are then combined to leverage both appearance and motion cues for tasks like action recognition.

## 3D ResNets(2017):

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/previous sota models/SOTA Models Residual block. Shortcut connections bypass a signal from the top of the block to the tail. Signals are summed at the tail..png" alt="Residual block. Shortcut connections bypass a signal from the top of the block to the tail. Signals are summed at the tail.">
</div>

Standard 3D CNNs extend the concept to simultaneously capture spatial and temporal information using 3D kernels (2D spatial information + temporal information). A drawback of this model is that the large number of parameters result the training being more computationally intensive and hence slower than the 2D version. Therefore, the 3D version of the ConvNets typically has fewer layers than the deeper architectures of 2D CNNs.

In this paper, the authors applied the ResNet architecture to the 3D CNNs. This approach introduces deeper models for 3D CNNs and achieves higher accuracy.

Experiments showed that the 3D ResNets (especially deeper ones like the ResNet-34) outperform models like the C3D, particularly on larger datasets. Pretrained models like Sports-1M C3D can help mitigate overfitting on smaller datasets. Overall, 3D ResNets effectively leverage deeper architectures to capture complex spatiotemporal patterns in the video data.

| Method                | Validation set |             |            | Testing set |             |            |
|-----------------------|----------------|-------------|------------|-------------|-------------|------------|
|                       | Top-1          | Top-5       | Average    | Top-1       | Top-5       | Average    |
| 3D ResNet-34   | 58.0           | 81.3        | **69.7**   | -           | -           | **68.9**   |
| C3D*                  | 55.6           | 79.1        | 67.4       | 56.1        | 79.5        | 67.8       |
| C3D w/ BN         | 56.1           | 79.5        | 67.8       | -           | -           | -          |
| RGB-I3D w/o ImageNet | -          | -           | 68.4       | 88.0        | **78.2**    |

## (2+1)D ResNets(2017):

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/previous sota models/SOTA Models 3D vs (2+1)D convolution..png" alt="3D vs (2+1)D convolution.">
</div>

(2+1)D ResNets are inspired by the 3D ResNets. However, a key difference lies in how the layers are structured. This architecture introduces a combination of 2D convolution and 1D convolution:

- The 2D convolution captures the spatial features within a frame.
- The 1D convolution captures the motion information across the consecutive frames.

This model can learn spatiotemporal features directly from video data, potentially leading to better performance in video analysis tasks like action recognition.

- Benefits:
    - The addition of nonlinear rectification (ReLU) between two operations doubles the number of non-linearities compared to a network using full 3D convolution for the same number of parameters, thus rendering the model capable of representing more complex functions.
    - Decomposition facilitates the optimization, yielding in lower train loss and test loss in practice.

| Method                        | Clip@1 Accuracy | Video@1 Accuracy | Video@5 Accuracy |
|-------------------------------|--------|---------|---------|
| DeepVideo                | 41.9   | 60.9    | 80.2    |
| C3D                     | 46.1   | 61.1    | 85.2    |
| 2D ResNet-152            | 46.5  | 64.6    | 86.4    |
| Conv pooling             | -      | 71.7    | 90.4    |
| P3D                      | 47.9  | 66.4   | 87.4   |
| R3D-RGB-8frame                | 53.8   | -       | -       |
| R(2+1)D-RGB-8frame            | 56.1   | 72.0    | 91.2    |
| R(2+1)D-Flow-8frame           | 44.5   | 65.5    | 87.2    |
| R(2+1)D-Two-Stream-8frame     | -      | 72.2    | 91.4    |
| R(2+1)D-RGB-32frame           | **57.0**   | **73.0**    | **91.5**    |
| R(2+1)D-Flow-32frame          | 46.4   | 68.4    | 88.7    |
| R(2+1)D-Two-Stream-32frame    | -      | **73.3**    | **91.9**    |

# Current Research

Currently, researchers are exploring deeper 3D CNN architectures. Another promising approach is combining 3D CNNs with other techniques like attention mechanisms. Alongside that, there is a push for developing larger video datasets like [Kinetics](https://github.com/google-deepmind/kinetics-i3d).
The Kinetics dataset is a large-scale high-quality video dataset commonly used for human action recognition research. It contains hundreds of thousands of video clips that cover a wide range of human activities.

<Tip>


</Tip>

# Conclusion

The evolution of video analysis models has been fascinating to witness. These models were heavily influenced by other SOTA models. For example, Two-StreamNets was motivated by the  ConvNets and (2+1)D ResNets were inspired by the 3D ResNets. As the research progresses, one can expect even more advanced architectures and techniques to emerge in the future.