import numpy as np
from matplotlib import pyplot as plt
import random
from Data_preparation import *

#--------------Visualize dataset
def plotImages(dataset):
    r = random.sample(dataset, 4)
    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.imshow((r[0])); plt.axis('off')
    plt.subplot(222)
    plt.imshow((r[1])); plt.axis('off')
    plt.subplot(223)
    plt.imshow((r[2])); plt.axis('off')
    plt.subplot(224)
    plt.imshow((r[3])); plt.axis('off')
    plt.show()

plotImages(list(faces_matrix.reshape(213, 64, 64)))

#---------------Plot Mean face--------------------

def mean_of_all_faces():
    plt.imshow(mean_face.reshape(64,64),cmap='gray'); 
    plt.title('Mean Face')
    
mean_of_all_faces()


#---------------Visualize first 10 PCs / eigenfaces-----------------

def plot_eigen_faces(eigen_vecs):
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    for i in np.arange(10):
        ax = plt.subplot(2,5,i+1)
        img = eigen_vecs[:,i].reshape(64,64)
        plt.imshow(img, cmap='gray')
    fig.suptitle("First 10 Eigenfaces", fontsize=16)
    plt.show()

plot_eigen_faces(eigen_vecs)


#----------plot some of reconstructed images with k = 1, 40, 120-----------

def plot_reconstructed_photos(faces_norm):
    
    for index in range(5):
        fig, axs = plt.subplots(1,3,figsize=(15,6))
        for k, i in zip([1,40,120],np.arange(3)):
            weight = faces_norm[index,:].dot(eigen_vecs[:,:k]) # Get PC scores of the images
            projected_face = weight.dot(eigen_vecs[:,:k].T) # Reconstruct first face in dataset using k PCs
            reconstructed_images = projected_face.reshape(64,64) + mean_face.reshape(64,64)
            ax = plt.subplot(1,3,i+1)
            ax.set_title("k = "+str(k))
            plt.imshow(projected_face.reshape(64,64)+ mean_face.reshape(64,64),cmap='gray');

        fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16);
        plt.show()
    return reconstructed_images

plot_reconstructed_photos(faces_norm)


#---------------2D and 3D plot-------------------------

def plot2D(projected):
    plt.figure(figsize=(14, 9))
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.title("2D plot")
    plt.show()
    
projected2D =  PCA(faces_norm, 2)
plot2D(projected2D)


from mpl_toolkits import mplot3d

def plot3D(projected):
    plt.figure(figsize=(14, 9))
    ax = plt.axes(projection='3d')
    ax.scatter(projected[:,0],projected[:,1],projected[:,2], cmap='viridis', linewidth=0.5);
    plt.show()

projected3D =  PCA(faces_norm, 3)
plot3D(projected3D)


#----------------------MSE------------------

#calculates MSE for one sample
def mse(original_imgs, rec):
    return np.power(np.subtract(original_imgs, rec), 2).mean(axis=None)

#--------------plots the MSE in terms of eigen vectors-------------------

def mse_analysis(std):
    x = []
    y = []
    idx = np.linspace(1, 630, 120)
    for i in idx:
        rec = reconstruction(std, k=int(i))
        rec = np.array(rec).reshape(213,4096)
        x.append(int(i))
        y.append(mse(std, rec))

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(14, 9))
    plt.ylabel('MSE')
    plt.xlabel('K = Number of eigenvectors (Reduced dim.)')
    plt.plot(x, y, label='MSE in terms of # of eigenvectors', alpha=0.8)
    plt.legend(loc='best')
    #plt.savefig('pca_mse')
    plt.show()

mse_analysis(faces_norm)

#---------------explained_variance---------------
def explained_variance():
    eig_vals_total = np.sum(eigen_Values[:120])
    var_exp = eigen_Values[:120] / eig_vals_total
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(var_exp.shape[0]), var_exp, alpha = 0.5, 
            align = 'center', label = 'individual explained variance')
    plt.step(range(var_exp.shape[0]), cum_var_exp, 
             where = 'mid', label = 'cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.ylim(0, 1.1)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.show()

explained_variance()
