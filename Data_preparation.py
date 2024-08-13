import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im

#-----------------Loading the whole image dataset, grayscale and resizing------------------
def load_images(path = None):
    neutral = []
    dir = path
    photos=os.listdir(dir)
    images=[dir+'/' + photo for photo in photos]

    for image in images:
        img = im.open(image).convert('L')
        img = img.resize((64,64))
        img2 = np.array(img).flatten() # vectorization
        neutral.append(img2)
    return neutral

faces_matrix = np.vstack(load_images('./jaffedbase'))
n_samples, n_features = faces_matrix.shape
     
#-----------------Normalization---------------------    
def normalize(faces):
    mean_face = np.mean(faces, axis=0)
    faces_norm = faces - mean_face
    return faces_norm, mean_face

faces_norm, mean_face = normalize(faces_matrix)



#--------------------Calculate covariance matrix---------------------------

face_cov = (np.dot(faces_norm.T, faces_norm)) / n_samples 


#-----------------Computing eigen_faces and eigen_values----------------------
def eigen_decomposition(cov_matrix):
    
    eigen_Values, eigen_vecs = np.linalg.eig(cov_matrix)
    idx = eigen_Values.argsort()[::-1]   
    eigen_Values = eigen_Values[idx]
    eigen_vecs = eigen_vecs[:,idx]
    eigen_vecs = eigen_vecs.astype('float64')
    
    return eigen_vecs, eigen_Values

eigen_vecs, eigen_Values = eigen_decomposition(face_cov)



#------------------PCA----------------



def PCA(X,components):

    X_pca = np.dot(X, eigen_vecs[:,:components])
    
    return X_pca


K = 1
projected_face = PCA(faces_norm,K)


K = 40
projected_face = PCA(faces_norm,K)

K = 120
projected_face = PCA(faces_norm,K)


#--------------Reconstruct--------------

def reconstruction(faces_norm,k):
    reconstructed_images = []
    weight = faces_norm.dot(eigen_vecs[:,:k])
    projected_face = weight.dot(eigen_vecs[:,:k].T) # Reconstruct first face in dataset using k PCs
    reconstructed_images = projected_face + mean_face
    return np.array(reconstructed_images)



