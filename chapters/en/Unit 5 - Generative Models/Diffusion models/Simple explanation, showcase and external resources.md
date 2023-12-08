# Extras

## Personalisation

Although diffusion models and GANs are able to create many new unique items, they aren't always able to create the specifc thing you need. Hence, you're left with having to fine tune a model, which usually requires a lot of data and compute. However, there are some techniques that can be used to personalise a model with just a few examples.

One such example is Dreambooth by Google research, which is a training technique that updates the entire diffusion model by training on just a few images of a subject or style. It works by associating a special word in the prompt with the example images. The detials on Dreambooth can be found [here](https://dreambooth.github.io/) and the hugging face Dreambooth training documentation can be found [here](https://huggingface.co/docs/diffusers/training/dreambooth).

Below is an example of Dreambooth being used to train on 4 images of a dog and some inference examples.
![dog](https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg)

From this example it's clear that the model has learnt that specific dog and is able to generate new images of that dog in diverse poses and backgrounds. Although the compute, data and time are an improvment, others have found more efficient ways to personalise a model.

This is where the current most popular method for this comes in which is Low Rank Adaption (LoRA). This method was initially developed to efficiently fine-tune Large Language models by Microsoft in this [paper](https://arxiv.org/abs/2106.09685). The main idea is factorise the weight update matrix into a 2 low rank matricies, which are optimised during training whilst the rest of the model is frozen. [Here](https://huggingface.co/docs/peft/conceptual_guides/lora) you can find a conceptual guide on how LoRA works. 

Now if we put those ideas together we can use LoRA to efficiently fine-tune a diffusion model on a few examples using Dreambooth. A tutorial Google Colab notebook on how to do this can be found [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb).

Due to the quality and efficiency of this method, many people have created their own LoRA parameters which many can be found on a website called [Civitai](https://civitai.com/models). Here you can downloads the LoRA weights which usually are in the range of 50-500MB and can be loaded into their corresponding diffusion model like so:

```python
model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model,torch_dtype=torch.float16)
pipe.load_lora_weights("lora_weights.safetensors")
pipe.fuse_lora(lora_scale=0.8)
```

This makes it quick to load a customised diffusion model and use it for inference, especially with how many models there are to choose from. Then if we want to remove the LoRA weights we can call `pipe.unfuse_lora()` which will return the model to its original state. As for the `lora_scale` parameter, this is a hyperparameter that controls how much the LoRA weights are used during inference. A value of 1.0 means the LoRA weights are fully used and a value of 0.0 means the LoRA weights are not used at all. The best value is often between 0.7 and 1.0 but it's worth experimenting with different values to see what works best for your use case.

## Guided Diffusion via ControlNet

Diffusion models have many ways in which they can be guided to create a desired ouput, such as prompts, negative prompt, guidance scale, inpaiting and many others. Here, we will focus on a method that has many variants and can be combined with all the other methods, which is called ControlNet. It was introduced in this [paper](https://arxiv.org/abs/2302.05543) by Stanford University. This method allows us to guide the diffusion model with an image which usually holds very specific information such as depth, pose, edges and many others. This allows for more consitency in the generated images which often is a problem with diffusion models.

ControlNet can be used in both text to image and image to image. Below is a text 2 image example using a ControlNet which was trained on edge detection conditioning, with the top left image being used as input.
Here we can see how all of the generated images have a very similar shape but with different colours. This is because the ControlNet is guiding the diffusion model to create images with the same shape as the input image.
 ![bird](https://github.com/lllyasviel/ControlNet/raw/main/github_page/p1.png)

For code to run ContrloNet with Stable diffusion XL refer to the official documentation (here)[https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetPipeline] but if you just want to test out some examples take a look at this Spaces demo (here)[https://huggingface.co/spaces/hysts/ControlNet-v1-1] which lets you try many types of ControlNets.










