# Using Unsupervised Cross-Domain Generation For Mustache Transplant

#Run
Note that at the begining of each file there is hardcoded paths for essential folders such as training data or pretrained weights, before running the files please make sure those path are accurate.
Also, all files tested only on CUDA, thus it might doesn't work on CPU.

To run the autoencoder run autoencoder.py.
For creating the data set, download CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and use the attributes to seperate male images from female images.

To run the simple architecture run simple_arch.py.
The program using the weights of the already trained autoencoder, make sure you first train the autoencoder. Mustache images dataset is provided in mustache_images folder, create mustachless images dataset using CelebA attributes.


