## Overview

### What is Hiera?

[Hiera](https://arxiv.org/abs/2306.00989) (Hierarchical Vision Transformer) is an architecture that achieves high accuracy without the need for specialized components found in other vision models. The authors propose pretraining Hiera with a strong visual pretext task to remove unnecessary complexity and create a faster and more accurate model.

![Hiera Architecture](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/hiera_architecture.png)

### From CNNs to ViTs

CNNs and hierarchical models are well-suited for computer vision tasks because they can effectively capture the hierarchical and spatial structure of visual data. These models use fewer channels but higher spatial resolution in the early stages to extract simpler features and more channels but lower spatial resolution in the later stages to extract more complex features.

![CNNs](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/CNN_architecture.webp)

On the other hand, Vision Transformers (ViTs) are more accurate, scalable, and architecturally simple models that took computer vision by storm when they were introduced. However, this simplicity comes at a cost: they lack this “vision inductive bias” (their architecture is not designed to work specifically with visual data).

Many efforts have been made to adapt ViTs, generally by adding hierarchical components to compensate for this lack of inductive bias in their architecture. Unfortunately, all of the resulting models turned out to be slower, bigger and more difficult to scale.

### Hiera's Approach: Pretraining Task is All You Need

Authors of the Hiera paper argue that a ViT model can learn spatial reasoning and perform well on computer vision tasks by using a strong visual pretext task called MAE and thus, they can remove unnecessary components and complexity from state-of-the-art multi-stage vision transformers to achieve greater accuracy and speed.

What components are the paper authors actually removing? To understand this we first have to introduce [MViTv2](https://arxiv.org/abs/2112.01526) which is the base hierarchical architecture from which Hiera is derived. MViTv2 learns multi-scale representations over its four stages: it starts by modeling low level features with a small channel capacity but high spatial resolution, and then in each stage trades channel capacity for spatial resolution to model more complex high-level features in deeper layers.

![MViTv2](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/mvitv2.png)

Instead of digging deep into MViTv2's key features (since it's not our main topic), we will breifly explain them in the next section to illustrate how researchers create Hiera by simplyfing this base architecture.

### Simplifying MViTv2

![Simplifying MViTv2](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/hiera_changes.png)

This table lists all the changes that authors made to MViTv2 in order to create Hiera and how each change affects accuracy and speed on images and video.

- **Replacing relative with absolute position embeddings**: MViTv2 swaps the absolute position embeddings from the original [ViT](https://arxiv.org/abs/2010.11929) paper with relative ones added to attention in each block. Authors undo this change because it added more complexity to the model and, as it can be seen in the table, these relative position embeddings are not necessary when training with MAE (both accuracy and speed improve with this change).
- **Removing convolutions**: Since the key idea of the paper is that a model can learn spatial biases by pretraining with a strong visual pretext task, removing convolutions, which are vision specific modules and add potentially unnecessary overhead seems to be an important change. Authors first replace every conv with a max pooling layer which decreases accuracy at first because of the huge impact it has on the image features. However, they realize that they can remove some of these extra max pooling layers, specifically the ones with a stride of 1 since they are basically applying a ReLU on every feature map. By doing so, authors nearly returned to the accuracy they had before, while speeding up the model by 22% for images and 27% for video.

### Masked Autoencoder
Masked Autoencoder (MAE) is an unsupervised training paradigm. As with any other autoencoder, it consists of encoding high dimensional data (images) into a lower dimension representation (embeddings) in such a way that this data can be decoded into the original high dimensional data again. However, the visual MAE technique consists of dropping a certain amount of patches (around 75%), encoding the rest of the patches, and then trying to predict the missing ones. This idea has been used extensively in recent years as a pre-training task for image encoders.

![MAE](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/hiera_images/mae.png)

## Why Does Hiera Matter?

In the era dominated by transformer models, there are still a lot of attempts to improve this simple architecture, adding the complexity of CNN to convert them into hierarchical models again. Although hierarchical models excel in computer vision, this study demonstrates that achieving hierarchical transformers doesn't necessitate intricate architectural modifications. Instead, a concentrated emphasis on the training task alone can yield simple, fast, and precise models.