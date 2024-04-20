# AI Yoga Trainer


## Data



## Models

We plan to train 4 different models:

1. Vanilla CNN
2. ResNet
3. MoveNet + Classifier
    
    This model involves using pretrained feature extractor called MoveNet to generate embeddings of a pose given an input image. These embeddings are processed and then used to train a classifier which outputs a pose. All experiments are recorded in Weights & Biases [here](https://wandb.ai/aml-experiments/movenet-yoga-classifier/reports/MoveNet-Classifier-Experiments--Vmlldzo3NTU3MTM4?accessToken=c3vqmq0hve1f2iev2344e1lk6tczwe5uqrjhawu0bsr305wnmm04rjix3xrwchua). Our best model is available [here](https://drive.google.com/file/d/133Mx1-G-tNZehncoFqFWVaot6wJcl-PB/view?usp=sharing). The training and evaluation notebook is called `train_movenet.ipynb`.

4. ViT


### Convolutional Newral Network (CNN)


`train_cnn.py` includes the training script for CNN models. The script requires several libraries that can be found in the environment config `cnn_env.yaml`. 

#### Dataset

To use this script, users need to download and extract the dataset to a directory. It is recommended to use Kaggle's CLI to do so via 
`kaggle datasets download -d akashrayhan/yoga-82`.

After extraction, users can specify the path to the dataset with the `--datadir` argument.

#### Experiments
The script allows user to experiment with different CNN architectures (ResNets and MobileNets) and hyperparameters (epochs, learning rate, weight decay, dropout rate, etc.) we have experimented with. 


#### Our results
The best model configuration is the pretrained MobileNetV3 with 0.7 dropout rate. Predicting the testing data, it achieves 0.74 top-1 accuracy, 0.94 top-5 accuracy, 0.74 F1 score, 0.76 precision, 0.76 recall and 0.99 ROC AUC.
Yoga, originating from ancient Indian philosophy, encompasses a holistic approach to physical and mental well-being, integrating postures, breathing techniques, and meditation. Central to the practice of yoga is the precise execution of various yoga poses or asanas, each designed to target specific muscle groups, enhance flexibility, and promote inner harmony. Traditionally, mastering these poses has relied heavily on the guidance of experienced instructors and years of dedicated practice. By leveraging state-of-the-art deep learning architectures, we aim to develop a robust system capable of accurately identifying and categorizing various yoga poses from input images or video streams. Such a system holds immense potential to revolutionize the way yoga is learned, taught, and practiced, democratizing access to expert guidance and personalized feedback regardless of geographical location or instructor availability.