# Using Stable Diffusion
This chapter introduces the building blocks of Stable Diffusion which is a generative artificial intelligence (generative AI) model that produces unique photorealistic images from text and image prompts. It originally launched in 2022 and was made possible thanks to a collaboration with
[Stability AI](https://stability.ai/), [RunawayML](https://runwayml.com/) and CompViz Group at LMU Munich following the [paper](https://arxiv.org/pdf/2112.10752.pdf).

What will you learn from this chapter?
- Basic componenet of Stable Diffusion
- How to use txt2img, img2img, inpainting

## What do we need for Stable Diffusion to work?
To make this section interesting we'll try to answer some questions to understand the basic componenet of Stable Diffusion process. 
We'll talk about each components in brief as they are already covered in our Diffusers course. Also, you can visit our previous section which talks about GANS and Diffusion models in details.

- What strategies does Stable Diffusion imploy to learn new stuff?
    - It uses forward and reverse process of diffusion models. In forward process we add gaussian noise to an image until all that remains is the random noise, usually we cannot identify the final noisified version of the image.
    - In reserve process, we have a learned neural network that is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

    Both of these process happens for finite number of steps T(as per DDPM paper T=1000). You begin the process at time t=0 by sampling a real image from your data distribution, and the forward process samples some noise from a Gaussian distribution at each time step t, which is added to the image of the previous time step. To get more mathematical intuition on this please read [HF Blog](https://huggingface.co/blog/annotated-diffusion) on Diffusion Models. 

- Since our images can be huge how can we compress it?

    When you have large images, they require more computing power to process. This becomes very noticeable in a specific operation known as self-attention. The bigger the image, the more calculations are needed, and these calculations increase very quickly (in a way mathematicians call "quadratically") with the size of the image.
For example, if you have an image that's 128 pixels wide and tall, it has four times more pixels than an image that's only 64 pixels wide and tall. Because of how self-attention works, dealing with this larger image doesn't just need four times more memory and computing power, it actually needs sixteen times more (since 4 times 4 equals 16). This makes it challenging to work with very high-resolution images, as they require a lot of resources to process. 
Latent diffusion models address the high computational demands of processing large images by using a Variational Auto-Encoder (VAE) to shrink the images into a more manageable size. The idea is that many images have repetitive or unnecessary information. A VAE, after being trained on a lot of data, can compress an image into a much smaller, condensed form. This smaller version still retains the essential features of the original image.

- How are we linking text and images since we are using prompts?

    As we know that during inference time, we can feed in the description of an image we'd like to see and some pure noise as a starting point, and the model does its best to 'denoise' the random input into something that matches the caption.
SD leverages a pre-trained transformer model based on something called CLIP. CLIP's text encoder was designed to process image captions into a form that could be used to compare images and text, so it is well suited to the task of creating 
useful representations from image descriptions. An input prompt is first tokenized (based on a large vocabulary where each word or sub-word is assigned a specific token) and then fed through the CLIP text encoder, producing a 768-dimensional
(in the case of SD 1.X) or 1024-dimensional (SD 2.X) vector for each token. To keep things consistent prompts are always padded/truncated to be 77 tokens long, 
and so the final representation which we use as conditioning is a tensor of shape 77x1024 per prompt.

- How can we add-in good inductive biases? 

    Since, we are trying to generate something new(for eg. a realistic pokemon), we need a way to go beyond the images we have seen before(for eg. a anime pokemon). That's where U-net and self-attention comes into picture. 
Given a noisy version of an image, the model is tasked with predicting the denoised version based on additional clues such as a text description of the image. 
Ok, so how do we actually feed this conditioning information into the UNet for it to use as it makes predictions? The answer is something called cross-attention. Scattered throughout the UNet are cross-attention layers. 
Each spatial location in the UNet can 'attend' to different tokens in the text conditioning, bringing in relevant information from the prompt. 


## How to use txt2img, img2img, in-painting using Diffusers
This section introduces useful usecases and how can we use perform these tasks using Diffusers library.
- Text to Image(txt2img)
The idea is to pass in the text prompt and it convert to the image. 

<iframe
	src="https://hysts-controlnet-v1-1.hf.space/"
	frameborder="0"
	width="850"
	height="450">
</iframe>

Using diffusers library you can get txt2img working in simple 2 steps. For eg. 
`
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", generator=generator).images[0]
image `


- Img2Img

- Inpainting