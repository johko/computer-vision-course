# Transfer Learing of multimodals models

In the preceding sections, we've delved into the fundamental concepts of multimodal models such as CLIP and its related counterparts. In this chapter, we will try to find out how you can use different types of multimodal models for your tasks.

There are several approaches to how you can use a multimodal model for your task:

1. Zero\few shot learing. Zero\few shot learning is the use of large pre-trained models with the ability to solve a problem that was not in the training data. These approaches can be useful when there is little labeled data for a task (5-10 examples). or there is none at all and will be discussed in more detail in Unit 11

2. Training the model from scratch. This is the only option if there are no pre-trained model weights available. In this approach, we initialize the weights of the model randomly (or a little more complicated, e.g He initialization) and train the model as usual. However, this approach demands substantial amounts of data for training.

3. Transfer learing. Transfer learning, unlike training from scratch, uses the weights of the pretrained model as initial weights.

In this chapter we will focus on the transfer learning approach. We will look at how it can be used for multimodal models, and also show how it can be used to solve some interesting problems, such as Visual Question Answering and Open-Set Object Detection.

## Transfer learing

More formally, transfer learning is a machine learning technique in which knowledge, representations or patterns that are obtained from solving one problem are reused to solve another, but similar problem.

In the context of deep learning, in transfer learning, when training a model for a particular task, we use the weights of another model (called a pretrained model) as the initial weights. The pre-trained model has typically been trained on a huge amount of data and has useful knowledge about the nature and relationships in that data. This knowledge is embedded in the weights of this model, and therefore if we use them as initial weights, we seem to transfer the knowledge embedded by the pretrained model into ours.

This approach has a number of advantages:

* Reducing the amount of computing resources. Since the pretrained model was trained on a huge amount of data over a long period of time, transfer learning requires much less computing resources for model convergence

* Reducing the size of labeled data. For the same reason, less data is required to achieve decent quality on the test sample

* Transfer of knowledge. In the fine-tuning process, the model can use knowledge from the pre-trained model, leading to better results.

## Transfer learning applications

We will look at the practical application of transfer learning in several tasks. You can select jupyter notebook from the table for the task you like and follow it

| Task        | Description                                                      | Model                                             | Notebook    |
| ----------- | ---------------------------------------------------------------- | ------------------------------------------------- | ----------- |
| Fine-tune CLIP             | Fine-tuning CLIP on a custom dataset                             | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | [CLIP notebook](https://) |
| VQA                        | Answering a question in natural <br/> language based on an image | [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) | [VQA notebook](https://) |
| Image-to-Text              | Describing an image in natural language                          | [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) | [Text 2 Image notebook](https://) |
| Open-set object detection  | Detect objects by natural language input                         |  [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | [Grounding DINO notebook](https://) |
| Assistant (GTP-4V like)    | Instruction tuning in the multimodal field                       | [LLaVA](https://github.com/haotian-liu/LLaVA) | [LLaVa notebook](https://) |
