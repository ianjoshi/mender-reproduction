from abc import ABC, abstractmethod

class ANN(ABC):
    @abstractmethod
    def build_graph(self, adata, k: int, include_self: bool) -> None:
         """
         Construct an approximate k-nearest neighbor (ANN) graph for the spatial coordinates
         stored in `adata.obsm['spatial']`.
         This method should compute approximate neighbors for each point in the dataset using
         an ANN algorithm and populate the resulting adjacency matrix into
         `adata.obsp['spatial_connectivities']` as a sparse matrix.
         
         Parameters:
         - adata (anndata.AnnData): AnnData object containing spatial coordinates in `adata.obsm['spatial']`.
         -  k (int): Number of neighbors to retrieve (excluding or including self depending on `include_self`).
         - include_self (bool): Whether to include each point itself in its list of neighbors.

        Returns:
        - None
        """
         pass
