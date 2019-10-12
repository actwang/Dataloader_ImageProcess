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
6. Return a pair of tensors, (imgs, labels), when dataloader.get batch() is called

Visualize results after these operations.
