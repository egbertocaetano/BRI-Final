import numpy as np
from sklearn.neighbors import NearestNeighbors

samples = [[0., 0., 0.], [1., 1., 0.], [1., 1., 1.]]
#neigh = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
neigh = NearestNeighbors(n_neighbors=3, algorithm='kd_tree')
neigh.fit(samples)

print(neigh.kneighbors([[1., 1., 1.], [0, 0, 0]]))
#nbrs = neigh.radius_neighbors([[0, 0, 1.3]], 0.4, return_distance=False)
#np.array(nbrs[0][0])