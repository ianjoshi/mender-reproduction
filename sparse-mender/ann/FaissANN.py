import faiss
import numpy as np
from scipy.sparse import csr_matrix
from .ANN import ANN

class FaissANN(ANN):
    def __init__(self, nlist=50):
        """
        FAISS-based Approximate Nearest Neighbor implementation using IVF.

        Parameters:
        - nlist (int): Number of Voronoi cells (clusters) for the IVF index.
        """
        self.nlist = nlist

    def build_graph(self, adata, k=15, include_self=True):
        """
        Build a spatial neighbor graph using FAISS IVF index.

        Parameters:
        - adata: AnnData object with spatial coordinates in .obsm['spatial']
        - k (int): Number of nearest neighbors
        - include_self (bool): Whether to include the point itself in the neighbor list
        """
        coords = adata.obsm['spatial'].astype(np.float32)
        n, dim = coords.shape

        # Build IVF index with L2 metric
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)

        index.train(coords)
        index.add(coords)
        index.nprobe = min(10, self.nlist)

        _, indices = index.search(coords, k + 1)

        row, col, data = [], [], []
        for i, neighbors in enumerate(indices):
            neighbors = [j for j in neighbors if j != -1]

            if not include_self:
                neighbors = [j for j in neighbors if j != i][:k]
            else:
                if i in neighbors:
                    neighbors.remove(i)
                neighbors = [i] + neighbors[:k - 1]  # explicitly include self

            for j in neighbors:
                row.append(i)
                col.append(j)
                data.append(1.0)

        adata.obsp['spatial_connectivities'] = csr_matrix(
            (data, (row, col)), shape=(n, n)
        )
