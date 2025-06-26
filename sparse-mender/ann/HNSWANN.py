import numpy as np
from scipy.sparse import csr_matrix
from .ANN import ANN

class HNSWANN(ANN):
    def __init__(self, space='l2'):
        """
        HNSW-based Approximate Nearest Neighbor implementation.

        Parameters:
        - space (str): Distance metric ('l2', 'ip', or 'cosine').
        """
        self.space = space

    def build_graph(self, adata, k=15, include_self=True):
        """
        Build a spatial neighbor graph using HNSW (via hnswlib).

        Parameters:
        - adata: AnnData object with spatial coordinates in .obsm['spatial']
        - k (int): Number of nearest neighbors
        - include_self (bool): Whether to include the point itself as a neighbor
        """
        import hnswlib  # Avoid multiprocessing pickling issues

        coords = adata.obsm['spatial'].astype(np.float32)
        dim = coords.shape[1]
        n = coords.shape[0]

        # Initialize HNSW index with stronger parameters for better recall
        index = hnswlib.Index(space=self.space, dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=32)
        index.add_items(coords)
        index.set_ef(max(k + 1, 32))  # ensures enough neighbors are searched

        labels, _ = index.knn_query(coords, k=k + 1)

        row, col, data = [], [], []
        for i, neighbors in enumerate(labels):
            # Convert to list and remove any invalid (-1) indices
            neighbors = [j for j in neighbors.tolist() if j != -1]

            if not include_self:
                neighbors = [j for j in neighbors if j != i][:k]
            else:
                if i in neighbors:
                    neighbors.remove(i)
                neighbors = [i] + neighbors[:k - 1]  # include self explicitly if needed

            for j in neighbors:
                row.append(i)
                col.append(j)
                data.append(1.0)

        adata.obsp['spatial_connectivities'] = csr_matrix(
            (data, (row, col)), shape=(n, n)
        )
