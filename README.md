# 02476-mlo-awesome-2

## Members
- Aimas Lund, s174435
- Antarlina Mukherjee, s210142
- Ion Cararus, s213209
- Mihai Nipomici, s184432

---

### Overall goal of the project
The goal of the project is to design an image classification model to classify RGB images of animals.

### Choice of framework
As this is a project about images, we have chosen to utilize the Pytorch Image Models framework.

### How to you intend to include the framework into your project
The Pytorch Image Models framework provides a plethora of pre-trained models, which we seek to utilize for classification of images. Furthermore, the models provided in this framework can also be used as a basis for further specialized models that we construct and train ourselves.

### What data are you going to run on (initially, may change)
We have chosen to use the CIFAR-10 data set, which is a data set containing 60.000 32x32 RGB images, and corresponding labels of different animals. This data set it chosen, as there a relatively few classes to select from, with a reasonable training data size and even distribution of images distributed on each class.

### What deep learning models do you expect to use
We expect to use some variations of convolutional neural networks to classify the images. To save time, we will seek to utilize the pre-trained models from the Pytorch Image Models as much as possible.
