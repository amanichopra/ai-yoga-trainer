# AI Yoga Trainer



## Convolutional Newral Network (CNN)


`train_cnn.py` includes the training script for CNN models. The script requires several libraries that can be found in the environment config `cnn_env.yaml`. 

### Dataset

To use this script, users need to download and extract the dataset to a directory. It is recommended to use Kaggle's CLI to do so via 
`kaggle datasets download -d akashrayhan/yoga-82`.

After extraction, users can specify the path to the dataset with the `--datadir` argument.

### Experiments
The script allows user to experiment with different CNN architectures (ResNets and MobileNets) and hyperparameters (epochs, learning rate, weight decay, dropout rate, etc.) we have experimented with. 


### Our results
The best model configuration is the pretrained MobileNetV3 with 0.7 dropout rate. Predicting the testing data, it achieves 0.74 top-1 accuracy, 0.94 top-5 accuracy, 0.74 F1 score, 0.76 precision, 0.76 recall and 0.99 ROC AUC.