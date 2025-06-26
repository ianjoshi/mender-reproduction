from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
from .ANN import ANN
import numpy as np

class AnnoyANN(ANN):
    def __init__(self, n_trees=10):
        """
        Initialize the Annoy-based Approximate Nearest Neighbor class.

        Parameters:
        - n_trees (int): Number of trees to use when building the Annoy index.
        """
        self.n_trees = n_trees

    def build_graph(self, adata, k=15, include_self=True):
        """
        Builds a sparse graph of spatial neighbor relationships using Annoy.

        Parameters:
        - adata: AnnData object with 'spatial' coordinates in .obsm
        - k (int): Number of nearest neighbors
        - include_self (bool): Whether to include the point itself as a neighbor

        Returns:
        - Updates adata.obsp['spatial_connectivities'] with a sparse CSR matrix
        """
        coords = adata.obsm['spatial'].astype(np.float32)
        dim = coords.shape[1]
        n = coords.shape[0]

        # Build index inside method to avoid pickling issues
        index = AnnoyIndex(dim, 'euclidean')
        for i, vec in enumerate(coords):
            index.add_item(i, vec.tolist())
        index.build(self.n_trees)

        row, col, data = [], [], []
        for i in range(n):
            neighbors = index.get_nns_by_item(i, k + 1)
            neighbors = [j for j in neighbors if j != -1]

            if not include_self:
                neighbors = [j for j in neighbors if j != i][:k]
            else:
                if i in neighbors:
                    neighbors.remove(i)
                neighbors = [i] + neighbors[:k - 1]  # explicitly add self if needed

            for j in neighbors:
                row.append(i)
                col.append(j)
                data.append(1.0)

        adata.obsp['spatial_connectivities'] = csr_matrix(
            (data, (row, col)), shape=(n, n)
        )
