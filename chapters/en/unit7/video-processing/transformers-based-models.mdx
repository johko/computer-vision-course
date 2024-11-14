# Transformers in Video Processing (Part 1)

## Introduction

In this chapter, we will cover how the Transformers model is utilized in video processing. In particular, we will introduce the Vision Transformer, a successful application of the Transformers model in the field of vision. We will then explain the additional considerations made for the Video Vision Transformer (ViViT) model used in video, as opposed to the Vision Transformer model used in images. Finally, we will briefly discuss about the TimeSFormer model. 

**Materials that would be helpful to review before reading this document**:

- [computer vision course / unit3 / vision transformers for image classification](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification)
- [transformers / model documentation: ViT](https://huggingface.co/docs/transformers/main/en/model_doc/vit)

## Recap about ViT

First, let's take a quick look at Vision Transformers: [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929), the most basic of the successful applications of Transformers to vision.

The abstract from the paper is as follows;

*Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.*

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_1_vit_architecture.png" alt="Vision transformer architecture">
</div>
<small>ViT architecture. Taken from the <a href= "https://arxiv.org/abs/2010.11929"> original paper</a>.</small>

The key techniques proposed in the ViT paper are as follows:

- Images are divided into small patches, and each patch is used as input to a Transformer model, replacing CNNs with a Transformer-based approach.  

- Each image patch is linearly mapped, and positional embeddings are added to allow the Transformer to recognize the order of the patches.  

- The model is pre-trained on large-scale datasets and fine-tuned for downstream vision tasks, achieving high performance.

### Performance & Limitation

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_2_vit_performance.JPG" alt="Vision transformer performance">
</div>
<small> Comparision with SOTA models. Taken from the <a href="https://arxiv.org/abs/2010.11929">original paper</a>.</small>

Although ViT outperformed other state-of-the-art models, training the ViT model required a large amount of computational power. Training the ViT model took 2,500 days on TPU-v3. Assuming a TPU-v3 core costs approximately $2 per hour (you can find more detailed pricing information [here](https://cloud.google.com/tpu/pricing)), it would cost $2 x 24 hours x 2,500 days = $120,000 to train the model once.

## Video Vision Transformer (ViViT)

As mentioned earlier, the important issue for ViViT, which extends the image processing of ViT to video classification task, was how to train the model more quickly and efficiently. Also, unlike images, video contains not only spatial information, but also temporal information, and how to handle this “temporal information” is a key consideration and exploration.

The abstract from the [paper](https://arxiv.org/abs/2103.15691) is as follows:

*We present pure-transformer based models for video classification, drawing upon the recent success of such models in image classification. Our model extracts spatiotemporal tokens from the input video, which are then encoded by a series of transformer layers. In order to handle the long sequences of tokens encountered in video, we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. Although transformer-based models are known to only be effective when large training datasets are available, we show how we can effectively regularise the model during training and leverage pretrained image models to be able to train on comparatively small datasets. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple video classification benchmarks including Kinetics 400 and 600, Epic Kitchens, Something-Something v2 and Moments in Time, outperforming prior methods based on deep 3D convolutional networks. To facilitate further research, we release code at https://github.com/google-research/scenic.*

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_3_vivit_architecture.png" alt="ViViT architecture">
</div>
<small>ViViT architecture. Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

### Embedding Video Clips

#### What is embedding?
Before diving into specific techniques, it's important to understand what embeddings are. In machine learning, embeddings are dense vector representations that capture meaningful features of input data in a format that neural networks can process. For videos, we need to convert the raw pixel data into these mathematical representations while preserving both spatial information (what's in each frame) and temporal information (how things change over time).

#### Why Video Embeddings Matter
Processing videos is computationally intensive due to their size and complexity. Good embedding techniques help by:

- Reducing dimensionality while preserving important features
- Capturing temporal relationships between frames
- Making it feasible for neural networks to process video data efficiently

#### Why Focus on Uniform Frame Sampling and Tubelet Embeddings?
These two techniques represent fundamental approaches in video processing that have become building blocks for more advanced methods:

1. They balance computational efficiency with information preservation, offering a range of options for different video processing tasks.
2. They serve as baseline methods, providing a comparison point against which newer techniques can demonstrate improvement.
3. Learning these approaches establishes a strong foundation in spatio-temporal processing, which is crucial for grasping more advanced video embedding methods.

#### Uniform Frame Sampling

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_4_uniform_sampling_1JPG.JPG" alt="Uniform frame sampling">
</div>
<small>Uniform Frame Sampling. Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

In this mapping method, the model uniformly samples some frames across the time domain,
e. g. one frame per every 2 frames. 

#### Tubelet Embedding

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_5_tubelet_embedding.JPG" alt="Tubelet embedding">
</div>
<small>Tubelet embedding. Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

An alternate method, extracting spatio-temporal "tubes" from the input volume and linearly projecting this. This method fuses spatio-temporal information during tokenization.

The previously introduced methods, such as Uniform Frame Sampling and Tubelet Embedding, are effective but relatively simple approaches. The upcoming methods to be introduced are more advanced.

### Transformer Models for Video in ViViT

The original ViViT paper proposes multiple transformer-based architectures, which we will now explore sequentially.

#### Model 1 : Spatio-Temporal Attention

The first model naturally extends the idea of ViT to the video classification task. Each frame in the video is split into  n_w(number of columns) x n_h(number of rows) image patches, resulting in a total of n_t(number of frames) x n_w x n_h patches. Each of these patches is then embedded as a “spatio-temporal token”—essentially a small unit representing both spatial(image) and temporal(video sequence) information. The model forwards all spatio-temporal tokens extracted from the video through the transformer encoder. This means each patch, or token, is processed to understand not only its individual features but also its relationship with other patches across time and space. Through this process, called “contextualizing,” the encoder learns how each patch relates to others by capturing patterns in position, color, and movement, thus building a rich, comprehensive understanding of the video’s overall context. 

**complexity : O(n_h^2 x n_w^2 x n_t^2)**

However, using attention on all spatio-temporal tokens can lead to heavy computational costs. To make this process more efficient, methods like Uniform Frame Sampling and Tubelet Embedding, as explained earlier, are used to help reduce these costs.

#### Model 2 : Factorised encoder

The approach in Model 1 was somewhat inefficient, as it contextualized all patches simultaneously. To improve upon this, Model 2 separates the spatial and temporal encoders sequentially.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_6_vivit_model2.JPG" alt="ViViT model 2">
</div>
<small>Factorised encoder (Model 2). Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

First, only spatial interactions are contextualized through a Spatial Transformer Encoder(=ViT). Then, each frame is encoded to a single embedding and fed into the Temporal Transformer Encoder(=general transformer).

**complexity : O(n_h^2 x n_w^2 + n_t^2)**

#### Model 3 : Factorised Self-Attention

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_7_vivit_model3.JPG" alt="ViViT model 3">
</div>
<small>Factorised Self-Attention (Model 3). Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

In model 3, instead of computing multi-headed self-attention across all pairs of tokens, we first only compute self-attention spatially (among all tokens extracted from the same temporal index). Next, we compute self-attention temporally (among all tokens extracted from the same spatial index). Because of the ambiguities, no CLS (classification) token is used.

**complexity : same as model 2**

#### Model 4 : Factorized dot-product attention

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_8_vivit_model4.JPG" alt="ViViT model 4">
</div>
<small>Factorised Dot-Product Attention (Model 4). Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

In model 4, half of the attention heads are designed to operate with keys and values from spatial indices, the other half operate with keys and values from same temporal indices.

**complexity : same as model 2, 3**

### Experiments and Discussion

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_9_vivit_performance.JPG" alt="ViViT model performance">
</div>
<small>Comparison of model architectures (Top 1 accuracy). Taken from the <a href = "https://arxiv.org/abs/2103.15691">original paper</a>.</small>

After comparing Models 1, 2, 3, and 4, it is evident that Model 1 achieved the best performance but required the longest training time. In contrast, Model 2 demonstrated relatively high performance with shorter training times compared to Models 3 and 4, making it the most efficient model overall.

 The ViViT model fundamentally faces the issue of dataset sparsity. Like the Vision Transformer(ViT), ViViT requires an extremely large dataset to achieve good performance. However, such a scale of dataset is often unavailable for videos. Given that the learning task is more complex, the approach is to first pre-train on a large image dataset using ViT to initialize the model. 

## TimeSFormer

TimeSFormer is a concurrent work with ViViT, applying Transformer on video classification. The following sections are explanations of each type of attention.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/transformer_based_video_model/unit7_10_timesformer.JPG" alt="TimeSFormer model">
</div>
<small>Visualization of the five space-time self-attention schemes. Taken from the <a href = "https://arxiv.org/abs/2102.05095">original paper</a>.</small>

- **Sparse Attention** is the same as ViT; the blue patch is the query and contextualizes other patches within one frame.
- **Joint Space-Time Attention** is the same as ViViT Model 1; the blue patch is the query and contextualizes other patches across multiple frames.
- **Divided Space-Time Attention** is similar to ViViT Model 3; the blue patch first contextualizes temporally with the green patches at the same position, and then spatially contextualizes with other image patches at the same time index.
- **Sparse Local Global Attention**: selectively combines local and global information.
- **Axial Attention**: processes spatial and temporal dimensions separately along their axes.

### Performance Discussion

The **Divided Space-Time Attention** mechanism shows the most effective performance, providing the best balance of parameter efficiency and accuracy on both K400 and SSv2 datasets.

## Conclusion

ViViT expanded upon the ViT model to handle video data more effectively by introducing various models such as the Factorized Encoder, Factorized Self-Attention, and Factorized Dot-Product Attention, all aimed at managing the space-time dimensions efficiently. Similarly, TimeSFormer evolved from the ViT architecture and utilized diverse attention mechanisms to handle space-time dimensions, much like ViViT. A key takeaway from this progression is the focus on reducing the significant computational costs associated with applying transformer architectures to video analysis. By leveraging different optimization techniques, these models improve efficiency and enable learning with fewer computational resources.

## Additional Resources

- [Video Transformers: A Survey](https://arxiv.org/abs/2201.05991)