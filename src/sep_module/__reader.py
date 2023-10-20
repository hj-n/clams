import pandas as pd

def read_clustme_data(path = "./clustme_data/"):
	"""
	read clustme scatterplots and human labels
	"""
	metadata = pd.read_csv(path + "metadata.csv")
	ids    			= metadata["XYposCSVfilename"].to_numpy()
	prob_single = metadata["probSingle"].to_numpy()
	clustme_data = []
	for i, curr_id in enumerate(ids):
		datum = pd.read_csv(path + "scatterplots/" + str(curr_id) + ".csv").to_numpy()
		clustme_data.append({
			"prob_single": prob_single[i],
			"data": datum
		})
	return clustme_data

