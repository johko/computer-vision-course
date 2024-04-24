# Image-based Joint-Embedding Predictive Architecture (I-JEPA)

## Overview

The Image-based Joint-Embedding Predictive Architecture (I-JEPA) is a groundbreaking self-supervised learning model [introduced by Meta AI in 2023](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/). It tackles the challenge of understanding images without relying on traditional labels or hand-crafted data augmentations.
To get to know I-JEPA better, let’s first discuss a few concepts.

### Invariance-based vs. Generative Pretraining Methods

We can say that there are broadly two main approaches for self-supervised learning from images: invariance-based methods and generative methods. Both approaches have their strengths and weaknesses.

- **Invariance-based methods**: In these methods, the model tries to reproduce similar embeddings for different views of the same image. And, of course, these different views are hand-crafted, the image augmentations we’re all familiar with. For example, rotating, scaling, and cropping. These methods are good at producing representations at high semantic levels, but the problem is that they introduce strong biases that may be detrimental to certain downstream tasks. For example, image classification and instance segmentation do not require data augmentations.

- **Generative methods**: The model tries to reconstruct the input image using these methods. That’s why these methods are sometimes called reconstruction-based self-supervised learning. Masks hide patches of the input image, and the model tries to reconstruct these corrupted patches at the pixel or token level (let’s keep this point in mind). This masked approach can easily generalize beyond image modality but doesn’t produce representations at the quality level of invariance-based methods. Also, these methods are computationally expensive and require large datasets for robust training.

Now let’s talk about Joint-Embedding Architectures.

### Joint-Embedding Architectures

This is a recent and popular approach for self-supervised learning from images in which two networks are trained to produce similar embeddings for different views of the same image. Basically, they train two networks to "speak the same language" about different views of the same picture. A common choice is the Siamese network architecture where the two networks share the same weights. But like everything else, it has its own problems:

- **Representation collapse**: A case in which the model produces the same representation regardless of the input.

- **Inputs compatibility criteria**: Finding good and appropriate compatibility measures can be challenging sometimes.

An example of a Joint-Embedding Architecture is [VICReg](https://arxiv.org/abs/2105.04906)

<Tip>

Different training methods can be employed to train Joint-Embedding Architectures, for example:

- Contrastive methods
- Non-Contrastive methods
- Clustering methods

</Tip>

So far so good, now to I-JEPA. As a start, the picture below from the I-JEPA paper shows the difference between Joint-Embedding methods, generative methods, and I-JEPA.

![I-JEPA Comparisons](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/i-jepa-1.png)

### Image-based Joint-Embedding Predictive Architecture (I-JEPA)

I-JEPA tries to improve on both generative and joint-embedding methods. Conceptually, it is similar to generative methods but with the following key differences:

1. **Abstract prediction**: This is the most fascinating aspect of I-JEPA in my opinion. Remember when we mentioned generative methods and how they try to reconstruct the corrupted input at the pixel level? Now, unlike generative methods, I-JEPA tries to predict it in representation space using its introduced predictor, that’s why they call it abstract prediction. This leads to the model learning more powerful semantic features.

2. **Multi-block masking**: Another design choice that improves the semantic features produced by I-JEPA is masking sufficiently large blocks of the input image.

### I-JEPA Components

The previous diagrams show and compare the I-JEPA architecture, below is a brief description of its main components:

1. **Target Encoder (y-encoder)**: Encodes target images and the target blocks are produced by masking its output.

2. **Context Encoder (x-encoder)**: Encodes a randomly sampled context block from the image to obtain a corresponding patch-level representation.

3. **Predictor**: Takes as input the output of the context encoder and a mask token for each patch we wish to predict and tries to predict the masked target blocks.

The target-encoder, context-encoder, and predictor all use a Vision Transformer (ViT) architecture. You have a refresher about them in unit 3 of this course.

The image below from the paper illustrates how I-JEPA works.

![I-JEPA method](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/i-jepa-2.png)

## Why It Matters

So, why I-JEPA? I-JEPA introduced many new design features while still being a simple and efficient method for learning semantic image representations without relying on hand-crafted data augmentations. Briefly,

1. I-JEPA outperforms pixel-reconstruction methods such as Masked autoencoders (MAE) on ImageNet-1K linear probing, semi-supervised 1% ImageNet-1K, and semantic transfer tasks.

2. I-JEPA is competitive with view-invariant pretraining approaches on semantic tasks and achieves better performance on low-level vision tasks such as object counting and depth prediction.

3. By using a simpler model with less rigid inductive bias, I-JEPA is applicable to a wider set of tasks.

4. I-JEPA is also scalable and efficient. Pretraining on ImageNet requires *less than 1200 GPU hours*.

## References

- [I-JEPA paper](https://arxiv.org/abs/2301.08243)

- [Meta's blog post about I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)

- [I-JEPA official GitHub repository](https://github.com/facebookresearch/ijepa)




