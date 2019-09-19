# Image Classifier

## Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Required Libraries](#libraries)
4. [Instructions](#instructions)
5. [Results](#results)


## Project Motivation <a name="motivation"></a>
The project goal is to build a Python application that can train an image classifier on a dataset, 
then predicts new images using the trained model. 
For this purpose, I built and trained an image classifier to recognize different species of flowers. 
I used [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

![Sample Flowers]('images/Flowers.png')

The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content


## File Descriptions <a name="files"></a>
- `Image Classifier Project.ipynb`: a Jupyter notebook, contains the whole project code to build and train an image classifier.
- `train.py`: trains a new network on a dataset and saves the model as a checkpoint.
- `predict.py`: uses a trained network to predict the class for an input image.
- `cat_to_name.json`: a JSON object which gives you a dictionary mapping the integer encoded categories
to the actual names of the flowers.


## Required Libraries <a name="libraries"></a>

- [PyTorch](https://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)


## Instructions <a name="instructions"></a>
- Train a new network on a data set with `train.py`
  - Basic usage: `python train.py data_directory`
  - Prints out training loss, validation loss, and validation accuracy as the network trains
  - Options:
    - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
    - Choose architecture: `python train.py data_dir --arch "vgg13"`
    - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
    - Use GPU for training: `python train.py data_dir --gpu`
    
- Predict flower name from an image with `predict.py` along with the probability of that name. 
That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.
  - Basic usage: `python predict.py /path/to/image checkpoint`
  - Options:
    - Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference: : `Use GPU for inference: python predict.py input checkpoint --gpu`
    
## Results <a name="results"></a>
- The testing accuracy of the trained network on the 10000 test images is 89%. 
- For sanity checking, I tested the trained network on the english marigold flower, 
and returned the top 5 classes.
As a result, it gave the correct class, the highest probability.

![Sample Flowers Output]('images/Flowers.png)
