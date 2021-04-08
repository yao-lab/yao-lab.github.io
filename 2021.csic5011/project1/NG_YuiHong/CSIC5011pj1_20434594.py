"""
This code the CSIC 5011 Project1 implementation on Manifold learning with hand written data
Author: CHENG Wei
Affiliation: ECE, HKUST
Contact: wchengad@connect.ust.hk

Code Reference: Sklearn tutorial https://scikit-learn.org/stable/modules/manifold.html
"""

print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.mixture import GMM

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
class Embedding:
    def __init__(self):
        pass
    
    def load_data(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.train_data = np.loadtxt(self.train_path)
        self.train_label = self.train_data[:,0]
        self.train_row = np.delete(self.train_data,0,1)
        self.train_sample_num, self.train_feature_num = self.train_row.shape
        self.train_image = np.reshape(self.train_row, [self.train_sample_num, 16, 16])

        self.test_data = np.loadtxt(self.test_path)
        self.test_label = self.test_data[:,0]
        self.test_row = np.delete(self.test_data,0,1)
        self.test_sample_num, self.test_feature_num = self.test_row.shape
        self.test_image = np.reshape(self.test_row, [self.test_sample_num, 16, 16])
 
    def set_param(self, n_components, n_neighbors):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def plot_embedding(self, X, y, image, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                    color=plt.cm.Set1(y[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_rows = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_rows) ** 2, 1)
                if np.min(dist) < 8e-3:
                    # don't show points that are too close
                    continue
                shown_rows = np.r_[shown_rows, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(image[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        # plt.show()
        plt.savefig(title,dpi=200)

    def svm_regression(self, train_X, test_X):
        clf = SVC()
        clf.fit(train_X, self.train_label)
        predict = clf.predict(test_X)
        score = clf.score(test_X, self.test_label) 
        return score, predict

    def logistic_regression(self, train_X, test_X):
        clf = LogisticRegression(random_state=0, solver='newton-cg', 
                                multi_class='multinomial')
        clf.fit(train_X, self.train_label)
        predict = clf.predict(test_X)
        score = clf.score(test_X, self.test_label) 
        return score, predict

    def knn_regression(self, train_X, test_X):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(train_X, self.train_label)
        predict = knn.predict(test_X)
        score = knn.score(test_X, self.test_label) 
        return score, predict

    # def gmm_regression(self, train_X, test_x):
    #     gmm = GMM(n_components=10, covariance_type='diag', init_params='wc', n_iter=20)
    #     gmm.fit(train_X, self.train_label)
    #     return gmm.score(test_X, self.test_label)
        
    def pca_embedding(self):
        print("Computing PCA embedding")
        pca = decomposition.TruncatedSVD(n_components=self.n_components)
        pca.fit(self.train_row)
        train_X = pca.transform(self.train_row)
        test_X = pca.transform(self.test_row)
        return train_X, test_X
    
    def isomap_embedding(self):
        print("Computing ISO map embedding")
        iso = manifold.Isomap(self.n_neighbors, n_components=self.n_components)
        iso.fit(self.train_row)
        train_X = iso.transform(self.train_row)
        test_X = iso.transform(self.test_row)
        return train_X, test_X
    
    def lle_embedding(self):
        print("Computing LLE embedding")
        lle = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=self.n_components,
                                            method='standard')
        lle.fit(self.train_row)
        train_X = lle.transform(self.train_row)
        test_X = lle.transform(self.test_row)
        return train_X, test_X

    def modified_lle_embedding(self):
        print("Computing modified LLE embedding")
        lle = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=self.n_components,
                                      method='modified')
        lle.fit(self.train_row)
        train_X = lle.transform(self.train_row)
        test_X = lle.transform(self.test_row)
        return train_X, test_X

    def lsta_embedding(self):
        lsta = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=self.n_components,
                                      method='ltsa')
        lsta.fit(self.train_row)
        train_X = lsta.transform(self.train_row)
        test_X = lsta.transform(self.test_row)
        return train_X, test_X
    
    def tsne_embedding(self):
        tsne = manifold.TSNE(n_components=self.n_components, init='pca', random_state=0)
        train_X = tsne.fit_transform(self.train_row)
        test_X = None
        return train_X, test_X

    def spectral_embedding(self):
        spc = manifold.SpectralEmbedding(n_components=self.n_components, random_state=0,
                                      eigen_solver="arpack")
        train_X = spc.fit_transform(self.train_row)
        test_X = None
        return train_X, test_X

    def run_embedding(self, embed_fn, regression_fn, plot=False, pred=False):
        train_X, test_X = embed_fn()
        if plot:
            self.plot_embedding(train_X[:, 0:2], self.train_label, self.train_image, embed_fn.__name__ + "_train.png")
            if test_X is not None:
                self.plot_embedding(test_X[:, 0:2], self.test_label, self.test_image, embed_fn.__name__ + "_test.png")
        if pred:
            totol_score, prediction = regression_fn(train_X, test_X)
            class_score = self.compute_class_score(prediction)
            # print(totol_score, class_score)
            return totol_score, class_score
    
    def compute_class_score(self, test_X):
        result = np.equal(test_X, self.test_label)
        score = []
        for i in range(0,10):
            mask = np.equal(self.test_label, np.ones_like(self.test_label) * i)
            s = sum((result * mask).astype(float)) / sum((mask).astype(float))
            score.append(s)
        return score

if __name__ == '__main__':
    em = Embedding()
    em.load_data(train_path="zip.train", test_path="zip.test")

    # visualize the embeddings in 2D
    em.set_param(n_components=2, n_neighbors=30)
    fns = [em.pca_embedding, em.isomap_embedding, em.lle_embedding, em.modified_lle_embedding, 
            em.lsta_embedding, em.tsne_embedding, em.spectral_embedding]
    for fn in fns:
        em.run_embedding(fn, em.knn_regression, plot=True, pred=False)

    # analyze the classification result
    em.set_param(n_components=20, n_neighbors=60)
    rgs = [em.svm_regression, em.knn_regression]
    fns = [em.pca_embedding, em.isomap_embedding, em.lle_embedding, em.modified_lle_embedding, 
            em.lsta_embedding]
    for rg in rgs:
        total_score_list = []
        class_score_list = []
        for fn in fns:
            total_score, class_score = em.run_embedding(fn, rg, plot=False, pred=True)
            total_score_list.append(total_score)
            class_score_list.append(class_score)
            print(fn.__name__, rg.__name__, total_score, class_score)

    # analyze the dimensionality on classification
    total_score_list = []
    for d in range(2,50):
        em.set_param(n_components=d, n_neighbors=60)
        total_score, class_score = em.run_embedding(em.pca_embedding, em.svm_regression, plot=False, pred=True)
        total_score_list.append(total_score)
        print(d)

    plt.plot(range(2,50), total_score_list, 'go--', linewidth=2, markersize=12)
    plt.show()
