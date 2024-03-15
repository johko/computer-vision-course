# Multimodal Text Generation (BLIP)
## Introduction

After the release of CLIP, the AI community recognized the immense potential of larger datasets and contrastive learning in deep learning. One significant development in this area is [BLIP (Bootstrapping Language-Image Pre-training)](https://arxiv.org/abs/2201.12086), which extends the capabilities of multimodal models to include text generation.

## CapFilt: Caption and Filtering
As multimodal models require large datasets, they often have to be scraped from the internet using image and alternative-text (alt-text) pairs. However, the alt-texts often do not accurately describe the visual content of the images, making them a noisy signal that is suboptimal for learning vision-language alignment. Hence, the BLIP paper introduced a caption and filtering mechanism (CapFilt). This is made up of a deep learning model which filters out noisy pairs and another model which creates captions for images. Both of these models are first fine-tuned using a human annotated dataset. They found that cleaning the dataset using CapFit produced superior performance to just using the web dataset. Further details on this process can be found in the [BLIP paper](https://arxiv.org/abs/2201.12086). 

## BLIP Architecture and Training
The BLIP architecture combines a vision encoder and a Multimodal Mixture of Encoder-Decoder (MED), enabling versatile processing of both visual and textual data. Its structure is shown in the figure below which features (blocks with the same color share parameters):

- **Vision Transformer (ViT):** This is a plain vision transformer featuring self-attention, feed-forward blocks, and a [CLS] token for embedding representation.
- **Unimodal Text Encoder:** Resembling BERT's architecture, it uses a [CLS] token for embedding and employs contrastive loss like CLIP, for aligning image and text representations.
- **Image-Grounded Text Encoder:** This substitutes the [CLS] token with an [Encode] token. Cross-attention layers enable the integration of image and text embeddings, creating a multimodal representation. It employs a linear layer to assess the congruence of image-text pairs.
- **Image-Grounded Text Decoder:** Replacing the bidirectional self-attention with causal self-attention, this decoder is trained via cross-entropy loss in an autoregressive manner for tasks like caption generation or answering visual questions.

BLIP's architecture integrates a vision encoder with a multimodal mixture of encoder-decoder components, enabling advanced text and image processing. This design allows it to adeptly handle diverse tasks, from aligning image-text pairs to generating captions and answering visual questions.

## Example Use Case: BLIP-2
Following BLIP's success, it's creator Salesforce introduced BLIP-2, an enhanced iteration. BLIP-2's advancements and capabilities are detailed in the [BLIP-2 paper](https://arxiv.org/abs/2301.12597) and the [Hugging Face documentation](https://huggingface.co/docs/transformers/model_doc/blip-2). Here, we utilize BLIP-2 for visual questioning answering.

```python
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: How many remotes are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(
    device, torch.float16
)
outputs = model.generate(**inputs)
text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)
```
This code snippet illustrates the application of BLIP-2 for visual question answering. Experiment with more complex queries or explore this functionality further using the provided Gradio app:

<iframe
	src="https://merve-BLIP2-with-transformers.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>

## Conclusion

BLIP marks a significant milestone in multimodal text generation, leveraging CLIP's strengths to create a robust model. It underscores the importance of dataset quality over quantity, contributing to the advancement of multimodal learning. For more details, refer to the [BLIP paper](https://arxiv.org/abs/2201.12086), [BLIP-2 paper](https://arxiv.org/abs/2301.12597), and the Hugging Face documentation for [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) and [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2).