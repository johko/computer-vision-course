# Control over Diffusion Models

## Dreambooth

Although diffusion models and GANs can generate many unique images, they can't always generate what you need exactly. Hence, you have to fine-tune a model, which usually requires a lot of data and computation. However, some techniques can be used to personalize a model with just a few examples.

One example is Dreambooth by Google Research, a training technique that updates the entire diffusion model by training on just a few images of a subject or style. It works by associating a special word in the prompt with the example images. The details on Dreambooth can be found in the [paper](https://dreambooth.github.io/) and the [Hugging Face Dreambooth training documentation](https://huggingface.co/docs/diffusers/training/dreambooth).

Below, you can see the results of Dreambooth being used to train on 4 images of a dog and some inference examples.
![Dreambooth Dog Example](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/teaser_static.jpg)
You can recreate the results following the Hugging Face documentation given above.

From this example, it's clear that the model has learned about that specific dog and can generate new images of that dog in diverse poses and backgrounds. Although computing, data, and time are improvements, others have found more efficient ways to customize a model.

This is where the current most popular method for this comes in, which is Low Rank Adaptation (LoRA). This method was initially developed to efficiently fine-tune Large Language Models by Microsoft in this [paper](https://arxiv.org/abs/2106.09685). The main idea is to factorise the weight update matrix into 2 low-rank matrices, which are optimized during training whilst the rest of the model is frozen. [Hugging face documentation](https://huggingface.co/docs/peft/conceptual_guides/lora) has a good conceptual guide on how LoRA works.

Now, if we put those ideas together we can use LoRA to efficiently fine-tune a diffusion model on a few examples using Dreambooth. A tutorial Google Colab notebook on how to do this can be found [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb).

Due to the quality and efficiency of this method, many people have created their own LoRA parameters which many can be found on a website called [Civitai](https://civitai.com/models) and [Hugging Face](https://huggingface.co/collections/multimodalart/awesome-sdxl-loras-64f9af6d5cce4f4e8f351466). 
For Civitai you can download the LoRA weights which usually are in the range of 50-500MB and in the case of Hugging Face version you can just load the model directly from the model hub.
Below is an example of how to load the LoRA weights in both cases and then fuse them with the model.

We can start with installing diffusers library.
```bash
pip install diffusers
````
We will initialize the `StableDiffusionXLPipeline` and load LoRA adapter weights.
```python
from diffusers import StableDiffusionXLPipeline
import torch

model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
pipe.load_lora_weights(
    "lora_weights.safetensors"
)  # if you want to install from a weight file
pipe.load_lora_weights(
    "ostris/crayon_style_lora_sdxl"
)  # if you wish to install a lora from a repository directly
pipe.fuse_lora(lora_scale=0.8)
```

This makes it quick to load a customised diffusion model and use it for inference, especially since there are a lot of models to choose from. Then, if we want to remove the LoRA weights, we can call `pipe.unfuse_lora()` which will return the model to its original state. As for the `lora_scale` parameter, this is a hyperparameter that controls how much the LoRA weights are used during inference. A value of 1.0 means the LoRA weights are fully used and a value of 0.0 means the LoRA weights are not used at all. The best value is often between 0.7 and 1.0 but it's worth experimenting with different values to see what works best for your use case.

You can try some of the Hugging Face LoRA models in this Gradio demo:
<iframe
	src="https://multimodalart-LoraTheExplorer.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>

## Guided Diffusion via ControlNet

Diffusion models have many ways in which they can be guided to create a desired output, such as prompts, negative prompts, guidance scale, inpainting and many others. Here, we will focus on a method that has many variants and can be combined with all the other methods, called ControlNet. It was introduced in this [paper](https://arxiv.org/abs/2302.05543) by Stanford University. This method allows us to guide the diffusion model with an image that usually holds very specific information such as depth, pose, edges, and many others. This allows for more consistency in the generated images, which is often a problem with diffusion models.

ControlNet can be used in both text-to-image and image-to-image. Below is a text 2 image example using a ControlNet which was trained on edge detection conditioning, with the top left image being used as input.
Here we can see how all of the generated images have a very similar shape but with different colours. This is because the ControlNet is guiding the diffusion model to create images with the same shape as the input image.
 ![bird](https://github.com/lllyasviel/ControlNet/raw/main/github_page/p1.png)

For code to run ControlNet with Stable Diffusion XL refer to the official documentation [here](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline) but if you just want to test out some examples take a look at this Gradio demo that lets you try different types of ControlNet:
<iframe
	src="https://hysts-ControlNet-v1-1.hf.space"
	frameborder="0"
	width="850"
	height="450">
</iframe>