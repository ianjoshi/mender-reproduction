import pysodb

class DataLoader:
    def __init__(self, dataset_name: str):
        """
        Initializes the DataLoader with the specified dataset name.
        
        Parameters:
        - dataset_name (str): The name of the dataset to load from SODB.
        """
        self.dataset_name = dataset_name
        self.sodb = pysodb.SODB()
        self.adata_dict = None

    def load(self):
        """
        Loads the dataset from SODB and stores it in adata_dict.
        """
        print(f"Loading dataset: {self.dataset_name}")
        self.adata_dict = self.sodb.load_dataset(self.dataset_name)
        print("Dataset loaded successfully.")
        return self.adata_dict

    def get_data(self):
        """
        Returns the loaded dataset, or None if it hasn't been loaded yet.
        """
        if self.adata_dict is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.adata_dict

if __name__ == "__main__":
    # Example usage
    dataset_name = 'Allen2022Molecular_aging'
    loader = DataLoader(dataset_name)
    adata_dict = loader.load()

    # Access the AnnData object and print summary
    adata = adata_dict.get('adata')
    if adata is not None:
        print(adata)
    else:
        print("No 'adata' object found in the dataset.")
