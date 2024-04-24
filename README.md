# AI Yoga Trainer

Yoga, originating from ancient Indian philosophy, encompasses a holistic approach to physical and mental well-being, integrating postures, breathing techniques, and meditation. Central to the practice of yoga is the precise execution of various yoga poses or asanas, each designed to target specific muscle groups, enhance flexibility, and promote inner harmony. Traditionally, mastering these poses has relied heavily on the guidance of experienced instructors and years of dedicated practice. By leveraging state-of-the-art deep learning architectures, we aim to develop a robust system capable of accurately identifying and categorizing various yoga poses from input images or video streams. Such a system holds immense potential to revolutionize the way yoga is learned, taught, and practiced, democratizing access to expert guidance and personalized feedback regardless of geographical location or instructor availability.

## Data

We leverage the Yoga-82 dataset, developed in a research [paper](https://arxiv.org/abs/2004.10362) titled “Yoga-82: A New Dataset for Fine-grained Classification of Human Poses”. The dataset boasts a hierarchical design to enable part-based learning where “the network [can] learn features based on the hierarchical classes”. There are a total of 82 poses in the dataset with an average of 347 images per class. The images are varied in their backgrounds and angles; some backgrounds are plain white while others are at a beach, indoors, or sketched. The class names are defined in English, though the Sanskrit counterparts are available as metadata. The hierarchy can be viewed in the appendix. 

The dataset can be downloaded from Kaggle using `kaggle datasets download -d akashrayhan/yoga-82`. Our EDA is conducted in `eda.ipynb` and the data is preprocessed and loaded into Google Cloud Storage for training in `load_data.ipynb`.


## Models

We train 4 different models:

1. ResNet
    
    Researchers have refined the architectures of convolutional neural networks to support deeper networks with more efficient training. In this project, we have experimented with several state-of-the-art deep convolutional neural networks, including ResNets of all sizes (18, 34, 50, 101, and 152) and MobileNets (V2 and V3). The pretrained models come from HuggingFace's community. `train_cnn.py` includes the training script (notebook version in `train_cnn.ipynb`) for these models and users can experiment with different architectures and hyperparameters. 
    
3. MoveNet + Classifier
    
    This model involves using pretrained feature extractor called MoveNet to generate embeddings of a pose given an input image. These embeddings are processed and then used to train a classifier which outputs a pose. All experiments are recorded in Weights & Biases [here](https://wandb.ai/aml-experiments/movenet-yoga-classifier/reports/MoveNet-Classifier-Experiments--Vmlldzo3NTU3MTM4?accessToken=c3vqmq0hve1f2iev2344e1lk6tczwe5uqrjhawu0bsr305wnmm04rjix3xrwchua). Our best model is available [here](https://drive.google.com/file/d/133Mx1-G-tNZehncoFqFWVaot6wJcl-PB/view?usp=sharing). The training and evaluation notebook is called `train_movenet.ipynb`.

4. ViT

    The Vision Transformer (ViT) is a novel machine learning model designed for image classification tasks, offering a departure from traditional Convolutional Neural Networks (CNNs). It achieves this through the use of self-attention mechanisms to capture global dependencies between image patches. ViT first divides the input image into fixed-size patches, which are then flattened into vectors and linearly projected into a lower-dimensional space. Special learnable embeddings (tokens) are added to these patch embeddings to encode positional information. The model then employs a Transformer encoder, consisting of multiple layers of self-attention and feedforward neural networks, to process the input sequence. Self-attention allows each token to attend to all other tokens, enabling the model to consider global context when making predictions. Finally, the output token representations are used for classification, typically by feeding the representation of the [CLS] token into a linear layer to predict the image class. ViT has demonstrated impressive performance on various image classification benchmarks, showcasing the potential of self-attention mechanisms in computer vision tasks. The training and evaluation script is called `train_vit.ipynb`. All experiments are recorded in Weights & Biases [here](https://wandb.ai/aml-experiments/vit-yoga-classifier).

