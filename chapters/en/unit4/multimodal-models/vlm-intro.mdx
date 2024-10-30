# Introduction to Vision Language Models

What will you learn from this chapter:
- A brief introduction to multimodality
- Introduction to Vision Language Models
- Various learning strategies
- Common datasets used for VLMs
- Downstream tasks and evaluation

## Our World is Multimodal 
Humans explore the world through diverse senses: sight, sound, touch, and scent. A complete grasp of our surroundings emerges by harmonizing insights from these varied modalities.
We think of modality, initially introduced in mathematics as distinct peaks, in a poetic way as: "With each modality, a unique part to play, Together they form our understanding array. A symphony of senses, a harmonious blend, In perception's dance, our world transcends." In pursuit of making a AI capable to understand the world, the field of machine learning seeks to develop models capable of processing and integrating data across multiple modalities. However, several challenges, including representation and alignment, must be addressed. Representation explores techniques to effectively summarize multimodal data, capturing the intricate connections among individual modality elements. Alignment focuses on identifying connections and interactions across all elements.
Also, it is crucial to acknowledge the inherent difficulties in handling multimodality:
- One modality may dominate others.
- Additional modalities can introduce noise.
- Full coverage over all modalities is not guaranteed.
- Different modalities can have complicated relationships.

Despite these challenges, the machine learning community has significantly progressed in developing these systems. This chapter delves into the fusion of vision and language, giving rise to Vision Language Models (VLMs). For a deeper understanding of multimodality, please refer to the preceding section of this Unit.


## Introduction 
Processing images to generate text, such as image captioning and visual question-answering, has been studied for many years which includes autonomous driving, remote sensing, etc. We also have seen a shift from traditional ML/DL training from scratch to a new learning paradigm including pre-training, fine-tuning and prediction, which has shown great benefit since in the traditional way we may need to collect huge amount of data, etc. 

Firstly proposed in [ULMFIT Paper](https://aclanthology.org/P18-1031.pdf) this paradigm involves:
- Pre-training the model with extensive training data.
- Fine-tuning the pre-trained model with task-specific data.
- Utilizing the trained model for downstream tasks such as classification.

In VLMs, we employ this paradigm and extend it to combine visual images, yielding the desired results. For instance, in 2021, OpenAI introduced a groundbreaking paper called [CLIP](https://openai.com/research/clip) which significantly influenced the adoption of this approach. CLIP utilizes an image-text contrastive objective, learning by bringing paired images and texts close while pushing others far away in the embedding space. This pre-training method allows VLMs to capture rich vision-language correspondence knowledge, enabling zero-shot predictions by matching the embeddings of any given images and texts. Notably, CLIP outperformed Imagenet models on out-of-distribution tasks. Further exploration of CLIP and similar models is covered in the next section of this chapter!


## Mechanism
To enable the functionality of Vision Language Models (VLMs), a meaningful combination of both text and images is essential for joint learning. How can we do that? One simple/common way is given image-text pairs:
- Extract image and text features using text and image encoders. For images it can be **CNN** or **transformer** based architectures.
- Learn the vision-language correlation with certain pre-training objectives.
    - The pre-training objective can be divided into three groups:
        - **contrastive** objectives train VLMs to learn discriminative representations by pulling paired samples close and pushing others faraway in the embedding space.
        - **generative** objectives to make VLMs learn semantic features by training networks to generate image/text data.
        - **alignment** objectives align the image-text pair via global image-text matching or local region-word matching on embedding space.
- With the learned vision-language correlation, VLMs can be evaluated on unseen data in a zero-shot manner by matching the embeddings of any given images and texts.

![Basic Mechanism of CLIP model](https://huggingface.co/datasets/hf-vision/course-assets/resolve/99ac107ade7fb89aae792f3655341528e64e1fbb/clip_paper.png)

Existing research predominantly focuses on enhancing VLMs from three key perspectives:
- collecting large-scale informative image-text data. 
- designing effective models for learning from big data. 
- designing new pre-training methods/objective for learning effective vision-language correlation.

VLM pre-training aims to pre-train a VLM to learn image-text correlation, targeting effective zero-shot predictions on visual recognition tasks which can be segmentation, classification, etc. 

## Strategies
We can [group](https://lilianweng.github.io/posts/2022-06-09-vlm/#no-training) VLMs based on how we leverage the two modes of learning.

- Translating images into embedding features that can be jointly trained with token embeddings.
    - In this method we fuse visual information into language models by treating images as normal text tokens and train the model on a sequence of joint representations of both text and images. Precisely, images are divided into multiple smaller patches and each patch is treated as one "token" in the input sequence. e.g. [VisualBERT](https://arxiv.org/abs/1908.03557), [SimVLM](https://arxiv.org/abs/2108.10904).

- Learning good image embeddings that can work as a prefix for a frozen, pre-trained language model.
    - In this method we don't change the language model parameters when adapting to handle visual signals. Instead we learn such an embedding space for images, such that it is compatible with the language model’s. e.g. [Frozen](https://arxiv.org/abs/2106.13884), [ClipCap](https://arxiv.org/abs/2111.09734).

- Using a specially designed cross-attention mechanism to fuse visual information into layers of the language model.
    - For enhanced fusion of visual information across language model layers, employ a tailored cross-attention fuse mechanism to balance text generation and visual input. Eg. [VisualGPT](https://arxiv.org/abs/2102.10407), etc.
- Combine vision and language models without any training.
    - More recent methods like [MAGiC](https://arxiv.org/abs/2205.02655) does guided decoding to sample next token, without fine-tuning.


## Common VLM Datasets
These are some of the common dataset used in VLM's present in [HF dataset](https://huggingface.co/docs/datasets/index).

**[MSCOCO](https://huggingface.co/datasets/HuggingFaceM4/COCO)** contains 328K images and each paired with 5 independent captions.

**[NoCaps](https://huggingface.co/datasets/HuggingFaceM4/NoCaps)** designed to measure generalization to unseen classes and concepts, where in-domain contains images portraying only COCO classes, near-domain contains both COCO and novel classes, and out-of-domain consists of only novel classes.

**[Conceptual Captions](https://huggingface.co/datasets/conceptual_captions)** contains 3 million pairs of images and captions, mined from the web and post-processed.

**[ALIGN](https://huggingface.co/blog/vit-align)**  a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset.

**[LAION](https://huggingface.co/collections/laion/openclip-datacomp-64fcac9eb961d0d12cb30bc3)** dataset consists of image-text-pairs. The image-text-pairs have been extracted from the Common Crawl web data dump and are from random web pages crawled between 2014 and 2021. This dataset was used in training Stable-Diffusion Models.


## Downstream Tasks and Evaluation
VLMs are getting good at many downstream tasks, including image classification, object detection, semantic segmentation, image-text retrieval, and action recognition while surpassing models trained traditionally. 

Generally the setup used for evaluating VLMs is zero-shot prediction and linear probing. Zero-shot prediction is the most common way to evaluate the VLMs, where we directly apply pre-trained VLMs to downstream tasks without any task-specific fine-tuning. 

In linear probing, we freeze the pre-trained VLM and train a linear classifier to classify the VLM-encoded embeddings to measure its representation. How do we evaluate these models? We can check how they perform on datasets, e.g. given an image and a question, the task is to answer the question correctly! We can also check how these models reason answer questions about the visual data. For this, the most common dataset used is [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/). 

Standard datasets like MSCOCO might be straightforward for a model to learn due to their distribution, which may not adequately demonstrate a model's capacity to generalize across more challenging or diverse datasets. In response, datasets like [Hateful Memes](https://arxiv.org/abs/2005.04790) are created to address this problem by understanding the models capability to an extreme by adding difficult examples ("benign confounders") to the dataset to make it hard which showed that multimodal pre-training doesn't work and models had a huge gap with human performance. 

![Winogrand Idea](https://huggingface.co/datasets/hf-vision/course-assets/resolve/99ac107ade7fb89aae792f3655341528e64e1fbb/winogrand_paper.png) 

One more such dataset called **Winoground** was designed to figure out how good CLIP actually is. **Figure Above** This dataset challenges us to consider if models, despite their impressive results, truly grasp compositional relationships like humans or if they're generalizing data. For example, earlier versions of Stable Diffusion and other text-to-image models, were not able to clearly count fingers. So, there's still lot of amazing work to be done to get the VLM's to the next stage! 

## What's Next?
The community is moving fast and we can see already lot of amazing work like [FLAVA](https://arxiv.org/abs/2112.04482) which tries to have a single "foundational" model for all the target modalities at once. This is one possible scenario for the future - modality-agnostic foundation models that can read and generate many modalities! But maybe we also see other alternatives developing, one thing we can say for sure is . there is an interesting future ahead. 

To capture more on these recent advances feel free to follow the HF's [Transformers Library](https://huggingface.co/docs/transformers/index), and [Diffusers Library](https://huggingface.co/docs/diffusers/index) where we try to add recent advances and models as fast as possible! If you feel like we are missing something important, you can also open an issue for these libraries and contribute code yourself.
