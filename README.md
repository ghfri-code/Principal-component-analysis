# PCA
**Dataset : jaffe**
In this project, you will compute a PCA from a set of images of faces, each of size 128 * 128 in RGB format. The database provides facial images of 30 different people. For simplicity, convert the images into gray scale and resize them into 64 * 64. Attention that, eigenfaces are sets of
eigenvectors which can be used to work with face recognition applications. Each eigenface, as we'll see in a bit, appears as an array vector of pixel intensities. 

We can use PCA to determine which eigenfaces contribute the largest amount of variance in our data, and to eliminate those that don't contribute much. This process lets us determine how many dimensions are necessary to recognize a face as 'familiar'

### Visualize the dataset.


### Visualize the reduced dataset using 2D and 3D plots.
### Reconstruct the original data by using Kprinciple components (Show reconstructed images of each individual for K=1,40,120).
### Plot the MSE between the original and reconstructed images in terms of number of
eigenvectors.
### Visualize some of the first principal components.
• How many principle components are enough so that you have acceptable reconstruction?
How do you select them?
