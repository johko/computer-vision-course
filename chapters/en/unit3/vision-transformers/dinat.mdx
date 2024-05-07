# Dilated Neighborhood Attention Transformer (DINAT)

![DINAT Architecture Diagram](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dinat_images/dina_comparison.png)
## Overview of Architecture

The Dilated Neighborhood Attention Transformer (DiNAT) is an innovative hierarchical vision transformer designed to enhance the performance of deep learning models, particularly in visual recognition tasks. Unlike traditional transformers, which employ self-attention mechanisms that may become computationally expensive as models scale, DiNAT introduces Dilated Neighborhood Attention (DiNA). DiNA extends a local attention mechanism called Neighborhood Attention (NA) by incorporating sparse global attention without additional computational overhead. This extension allows DiNA to capture more global context, expand the receptive field exponentially, and model longer-range inter-dependencies efficiently. 

DiNAT combines both NA and DiNA in its architecture, resulting in a transformer model capable of preserving locality, maintaining translational equivariance, and achieving significant performance boosts in downstream vision tasks. The experiments conducted with DiNAT demonstrate its superiority over strong baseline models such as NAT, Swin, and ConvNeXt across various visual recognition tasks.

## Core of DiNAT: Neighborhood Attention

DiNAT is based on Neighborhood Attention (NA) architecture, an attention mechanism specifically designed for computer vision tasks, aiming to capture relationships between pixels in an image efficiently. In a simple analogy, imagine you have an image, and each pixel in that image needs to understand and focus on its nearby pixels to make sense of the entire picture. Let's examine the key features of NA:

- **Local Relationships**: NA captures local relationships, allowing each pixel to consider information from its immediate surroundings. This is similar to how we might understand a scene by looking at the objects closest to us first before considering the entire view.

- **Receptive Field**: NA allows pixels to grow their understanding of their surrounding without needing too much extra computation. It dynamically expands their scope or "attention span" to include more distant neighbors when necessary.

Essentially, Neighborhood Attention is a technique that enables pixels in an image to focus on their surroundings, helping them understand local relationships efficiently. This localized understanding contributes to building a detailed understanding of the entire image while managing computational resources efficiently.

![DiNAT Architecture Diagram](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/dinat_images/dinat_architecture.png)

## The Evolution of DINAT 

The development of the Dilated Neighborhood Attention Transformer represents a significant improvement in vision transformers. It addresses the limitations of existing attention mechanisms. Initially, Neighborhood Attention was introduced to provide locality and efficiency, but it fell short in capturing global context. To overcome this limitation, the concept of Dilated Neighborhood Attention (DiNA) was introduced. DiNA extends the NA by expanding neighborhoods into larger sparse regions. This allows for the capture of more global context and exponentially increases the receptive field without adding computational burden. The next development is DiNAT, which combines localized NA with the expanded global context of DiNA. DiNAT achieves this by gradually changing dilation throughout the model, optimizing receptive fields, and simplifying feature learning.

## Image Classification with DiNAT
You can classify among ImageNet-1k images using [shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-small-in1k-224) model with 🤗 transformers. You can also fine-tune it for your own use case.

```python
from transformers import AutoImageProcessor, DinatForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
model = DinatForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## References
- [DiNAT Paper](https://arxiv.org/abs/2209.15001) [1] 
- [Hugging Face DiNAT Transformer](https://huggingface.co/docs/transformers/model_doc/dinat) [2]  
- [Neighborhood Attention(NA)](https://arxiv.org/abs/2204.07143) [3] 
- [SHI Labs](https://huggingface.co/shi-labs) [4]  
- [OneFormer Paper](https://arxiv.org/abs/2211.06220) [5]
- [Hugging Face OneFormer](https://huggingface.co/docs/transformers/main/en/model_doc/oneformer) [6]  
