# Losses

## Introduction

Before diving into the different losses used to train models like CLIP, it is important to have a clear understanding of what contrastive learning is. Contrastive Learning is an unsupervised deep learning method aimed at representation learning. Its objective is to develop a data representation where similar items are positioned closely in the representation space and dissimilar items are distinctly separated.

In the image below, we have an example where we want to keep the representation from dogs closer to other dogs, but also far from cats.
![image info](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/contrastive_learning.png)

## Training objectives

### Constrative Loss

Contrastive loss is one of the first training objectives used for contrastive learning. It takes a pair of samples as input that can be similar or dissimilar, and the objective is to map similar samples close in the embedding space and push dissimilar samples apart.

Technically speaking, imagine that we have a list of input samples \\(x_n\\) from multiple classes. We want a function where examples from the same class have their embeddings close in the embedding space, and examples from different classes are far apart. Translating this to a mathematical equation, what we have is:

$$L = \mathbb{1}[y_i = y_j]||x_i - x_j||^2 + \mathbb{1}[y_i \neq y_j]max(0, \epsilon - ||x_i - x_j||^2)$$

Explaining in simple terms:

- If the samples are similar \\(y_i = y_j\\), then we minimize the term \\(||x_i - x_j||^2\\) that corresponds to their Euclidean distance, i.e., we want to make them closer;
- If the samples are dissimilar \\((y_i \neq y_j)\\), then we minimize the term \\( max(0, \epsilon - ||x_i - x_j||^2)\\) that is equivalent to maximizing their euclidean distance until some limit \\(\epsilon\\), i.e., we want to make them distant from each other.

## References

- [An Introduction to Contrastive Learning](https://www.baeldung.com/cs/contrastive-learning)
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)