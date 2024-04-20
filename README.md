# AI Yoga Trainer

Yoga, originating from ancient Indian philosophy, encompasses a holistic approach to physical and mental well-being, integrating postures, breathing techniques, and meditation. Central to the practice of yoga is the precise execution of various yoga poses or asanas, each designed to target specific muscle groups, enhance flexibility, and promote inner harmony. Traditionally, mastering these poses has relied heavily on the guidance of experienced instructors and years of dedicated practice. By leveraging state-of-the-art deep learning architectures, we aim to develop a robust system capable of accurately identifying and categorizing various yoga poses from input images or video streams. Such a system holds immense potential to revolutionize the way yoga is learned, taught, and practiced, democratizing access to expert guidance and personalized feedback regardless of geographical location or instructor availability.

## Data



## Models

We plan to train 4 different models:

1. Vanilla CNN
2. ResNet
3. MoveNet + Classifier
    
    This model involves using pretrained feature extractor called MoveNet to generate embeddings of a pose given an input image. These embeddings are processed and then used to train a classifier which outputs a pose. All experiments are recorded in Weights & Biases [here](https://wandb.ai/aml-experiments/movenet-yoga-classifier/reports/MoveNet-Classifier-Experiments--Vmlldzo3NTU3MTM4?accessToken=c3vqmq0hve1f2iev2344e1lk6tczwe5uqrjhawu0bsr305wnmm04rjix3xrwchua). Our best model is available [here](https://drive.google.com/file/d/133Mx1-G-tNZehncoFqFWVaot6wJcl-PB/view?usp=sharing). The training and evaluation notebook is called `train_movenet.ipynb`.

4. ViT