# Dataloader_ImageProcess
dataloader for image processing that read images, random resize, crop and flip, and generate labels for images
Tasks:
1. Read images in grayscale from a provided image path containing images at 600dpi
2. Randomly resize an image to one of the following resolutions: 600dpi, 500dpi,
   400dpi, and then resize it back to its original size (for 600dpi, use the original
   image directly).
   
3. Randomly crop the resulting image to size of 256 × 256.
4. Randomly flip the image horizontally or vertically
5. Generate the label for the images, which is the one-hot encoding of their resolution
   class index. Here we can use indices [0, 1, 2] to represent [600dpi, 500dpi, 400dpi],
   respectively.
6. Return a pair of tensors, (imgs, labels), when dataloader.get_batch() is called

Visualize results after these operations.

# diff_600 dpi
diff_600.py is a part of exploring the difference between the provided original high resolution image at 600 dpi and the image that is resized to the same resolution while being lower in resolution in the beginning. 
Compilation is as normal except need to change directory to where the images are stored. 

# model_or.py
**prerequisites**

* Import as written
* Have generated the image patches to corresponding directories (see generate_patches.py)
* Have LeNet5 under the same directory (model to train)
* Have dataloader python file under same directory
**Purpose**
This file contains the main training and evaluation process that I personnally used which reaches around 72% accuracy. 
The train_mode function is adopted from Pytorch official website tutorial(https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) with some modifications on how the dataloader was fetching the data. 
For the main part of the code, we adjusted some epoch numbers in the final model that we use (in other teammates' code) to ensure optimal training performance for our project. 

# generate_patches.py
**prerequisites**

* Import as written
* Have the text images stored
**detail**
this file contains a procedure and three functions needed: 
* train_gen_patch()
* valtest_gen_patch()
* is_blank()
1. For images that are used for training, we use 256 by 256 filters to crop them with step of 128. Storing these image patches under the patches folder under the training label. 
2. For validation and testing images, we use 128 by 128 filters to crop them with step of 128. Storing these image patches under the patches folder under val/testing label.
3. is_blank() basically checks for whether the image contains text by using a certain threshold determined by [testThreshold.py](/testThreshold.py)
# opt_resolut.py 
This file is our latest user side execution of taking a sample input image and giving its optimal resolution. It adopted some of the other functions included in [generate_patches.py](/generate_patches.py). And used pre-trained model and max pooling to get the final result.

# dataloader.py
This is our custom dataloader that have get_batch method which is primarily used in [model training](/model_or.py). It takes an extra argument 'state' which determines what type of transformation is going to be applied to the image. It returns a pair of tensors: imgs, labels. These are the input and ground truth that we will later use in [training process](/mode_or.py).

# LeNet.py
Modified original [LeNet5](https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py) to fit our training needs (mostly to fit our image patch size of 128x128). Pipeline description is in comments included in [code](/LeNet.py).
