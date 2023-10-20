from sklearn.mixture import GaussianMixture
import __reader as rd
import matplotlib.pyplot as plt
import numpy as np

import json
import __visualize as vg

from __helpers import decompose_covariance_matrix

import os


"""
Read ClustMe dataset and extract the gaussian info
The extracted info is used as a basis for the features used in the regression module
"""


def extract_single(datum):
	"""
	extract the gaussian mixture labels and params from a single clustme output file
	"""
	gmm = GaussianMixture(n_components=2, covariance_type='full')
	labels = gmm.fit_predict(datum)

	proba = gmm.predict_proba(datum)
	proba_based_label = np.random.rand(len(proba)) < proba[:, 0]
	proba_based_label = proba_based_label.astype(int)

	covariances = gmm.covariances_
	decompose_0 = decompose_covariance_matrix(covariances[0])
	decompose_1 = decompose_covariance_matrix(covariances[1])

	return gmm, {
		"means": gmm.means_.tolist(),
		"covariances": covariances.tolist(),
		"labels": labels.tolist(),
		"proba_labels": proba_based_label.tolist(),
		"weights": gmm.weights_.tolist(),
		"scaling": [decompose_0[0], decompose_1[0]],
		"rotation": [decompose_0[1], decompose_1[1]],
		"rotation_degree": [decompose_0[2], decompose_1[2]]
	}

def scale_datum(datum):
	"""
	scale the datum to the range of [0, 1]
	"""
	min_x = min(datum[:, 0])
	max_x = max(datum[:, 0])
	min_y = min(datum[:, 1])
	max_y = max(datum[:, 1])
	range_x = max_x - min_x
	range_y = max_y - min_y
	min_for_scale = min_x if range_x > range_y else min_y
	max_for_scale = max_x if range_x > range_y else max_y
	scaled_datum = (datum - min_for_scale) / (max_for_scale - min_for_scale)
	return scaled_datum


def extract(is_draw=False):
	"""
	extract the gaussian info and store it with original data in json format
	"""
	data = rd.read_clustme_data()

	os.makedirs("./clustme_processing/plot_individual", exist_ok=True)
	os.makedirs("./clustme_processing/extracted", exist_ok=True)

	for i, datum in enumerate(data):
		datum["data"] = scale_datum(datum["data"])
		gmm, gaussian_info = extract_single(datum["data"])
		if is_draw:
			vg.plot_gmm(gmm, datum["data"], gaussian_info["proba_labels"])
			plt.savefig("./clustme_processing/plot_individual/" + str(i) + ".png")
			# plt.savefig("./plot_individual_with_score/" + str(datum["prob_single"]) + ".png")
			plt.clf()
		datum["gaussian_info"] = gaussian_info
		datum["data"] = datum["data"].tolist()
		with open("./clustme_processing/extracted/" + str(i) + ".json", "w") as f:
			json.dump(datum, f)



if __name__ == "__main__":
	extract(is_draw=True)