# Retention In Vision

## What are Retention Networks
Retentive Network (RetNet) is a foundational architecture proposed for large language models in the paper [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621). This architecture is designed to address key challenges in the realm of large-scale language modeling: training parallelism, low-cost inference, and good performance.

![LLM Challenges](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/LLM%20Challenges.png)
RetNet is able to tackle these challenges by introducing the Multi-Scale Retention (MSR) mechanism, which is an alternative to the multi-head attention mechanism commonly used in Transformer models. 
MSR has a dual form of recurrence and parallelism, so it is possible to train the models in a parallel way while recurrently conducting inference. We will explore RetNet in detail in the later chapter.

The Multi-Scale Retention mechanism operates under three computation paradigms:
- **Parallel Representation:** This aspect of RetNet is designed similar to self-attention in Transformer, where it enables us train the models with GPUs efficiently.

- **Recurrent Representation:** This representation facilitates efficient inference with O(1) complexity in terms of memory and computational requirements. It significantly reduces deployment costs and latency, and simplifies implementation by eliminating the need for key-value cache strategies often used in traditional models.

- **Chunkwise Recurrent Representation:** This third paradigm addresses the challenge of long-sequence modeling. It achieves this by encoding each local block in parallel for computational speed while simultaneously and recurrently encoding global blocks to optimize GPU memory usage.

During the training phase, the approach incorporates both parallel and chunkwise recurrent representations, optimizing GPU usage for fast computation and being particularly effective for long sequences in terms of computational efficiency and memory use. 
For the inference phase, the recurrent representation is used, favoring autoregressive decoding. This method efficiently reduces memory usage and latency, maintaining equivalent performance outcomes.

## From Language to Image
### RMT
The paper [RMT: Retentive Networks Meet Vision Transformers](https://arxiv.org/abs/2309.11523) proposes a new vision backbone inspired by the RetNet architecture. The authors propose RMT to enhance the Vision Transformer (ViT) by introducing explicit spatial priors and reducing computational complexity, drawing inspiration from the RetNet's parallel representation. 
This includes adapting the RetNet’s temporal decay to spatial domains and using a [Manhattan distance-based](https://en.wikipedia.org/wiki/Taxicab_geometry) spatial decay matrix, along with an attention decomposition form, to improve efficiency and scalability in vision tasks.

- **Manhattan Self-Attention (MaSA)**
![Attention Comparison](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/Attention%20Comparison.png)
MaSA incorporates Self-Attention mechanism with a two-dimensional bidirectional spatial decay matrix based on the Manhattan distance among the tokens. This matrix decreases attention scores for tokens further away from a target token, allowing it to perceive global information while varying attention based on distance.

- **Decomposed Manhattan Self-Attention (MaSAD)**
![MaSAD](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MaSAD.png)
This mechanism decomposes Self-Attention in images along horizontal and vertical axes of the image, maintaining the spatial decay matrix without losing prior information. This decomposition allows the Manhattan Self-Attention (MaSA) to model global information efficiently with linear complexity, while preserving the original MaSA's receptive field shape.

However, unlike the original RetNet, which performs training with parallel representation and inference with recurrent representation, RMT does both processes with the MaSA mechanism. The authors have made comparisons between MaSA and other RetNet's representations, and they show that MaSA has the best throughput with the highest accuracy.
![MaSA vs Retention](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/MaSA%20vs%20Retention.png)

### ViR
![ViR](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/ViR.png)

Another work inspired by the RetNet architecture is the ViR, as discussed in the paper [ViR: Vision Retention Networks](http://arxiv.org/abs/2310.19731). In this architecture, the authors propose a general vision backbone with a redesigned retention mechanism. They demonstrate that ViR can scale favorably to larger image resolutions in terms of image throughput and memory consumption by leveraging the dual parallel and recurrent properties of the retentive network.

The overall architecture of ViR is quite similar to that of ViT, except that it replaces the Multi-Head Attention (MHA) with Multi-Head Retention (MHR). This MHR mechanism is free of any gating function and can be switched between parallel, recurrent, or chunkwise (a hybrid between parallel and recurrent) modes. Another difference in ViR is that the positional embedding is first added to the patch embedding, and then the [class] token is appended.

## Further Reading

- [RetNet's official repo](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py)
- [RetNet's Multi-Scale Retention official repo](https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py)
- [Retentive Networks (RetNet) Explained: The much-awaited Transformers-killer is here](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8)
- [Retentive Network: A Successor to Transformer for Large Language Models (Paper Explained)](https://www.youtube.com/watch?v=ec56a8wmfRk)
- [RMT's official repo](https://github.com/qhfan/RMT)
- [ViR's official repo](https://github.com/NVlabs/ViR)
