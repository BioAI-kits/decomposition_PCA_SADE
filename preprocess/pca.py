from sklearn.decomposition import PCA
import numpy as np

def decomposition_pca(data, n_components, save_file='./result/matrix_pca.txt'):
    """ 
    data: 
    n_components: dimension number return by PCA
    
    """
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    # print("PCA explained variance is : \n", pca.explained_variance_)
    
    np.savetxt(save_file, data_pca)
    
    return data_pca
    