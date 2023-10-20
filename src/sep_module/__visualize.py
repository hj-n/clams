import matplotlib.pyplot as plt
import __reader as rd
import numpy as np 
from matplotlib.patches import Ellipse

def visualize_clustme():
	clustme_data = rd.read_clustme_data()

	subplot_num = 10
	block_size  = subplot_num ** 2
	block_num   = int(len(clustme_data) / block_size)

	cmap = plt.get_cmap('coolwarm')
	for i in range(block_num):
		fig, axs = plt.subplots(subplot_num, subplot_num)
		fig.set_facecolor('black')
		fig.set_size_inches(60, 60)
		for j in range(subplot_num):
			for k in range(subplot_num):
				axs[j][k].scatter(
					clustme_data[i * block_size + j * subplot_num + k]["data"][:, 0], 
					clustme_data[i * block_size + j * subplot_num + k]["data"][:, 1],
					c=cmap(clustme_data[i * block_size + j * subplot_num + k]["prob_single"])
				)
				axs[j][k].axis("off")
				
	
	  ## save the current block figure
		fig.savefig("./plot/clustme_" + str(i) + ".png")
		
		plt.clf()



def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, labels, label=True, ax=None):
    ax = ax or plt.gca()
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=3, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=3, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)
				
    return ax


def plot_gmm_graph(gmm, X, labels, label=True, means=None, edges=None ,ax=None):
		ax = plot_gmm(gmm, X, labels, label, ax)

		for edge in edges:
			edge = edge.split("_")
			datum = np.array([means[int(edge[0])], means[int(edge[1])]])
			ax.plot(*datum.T, c="black")
