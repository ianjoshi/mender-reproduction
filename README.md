# MENDER Reproduction & Exploration
This repository contains a reproduction of key figures and additional research based on the original MENDER paper:

> **MENDER: Fast and scalable tissue structure identification in spatial omics data**  
> Zhiyuan Yuan et al., Nature Communications, 2024  
> [DOI: 10.1038/s41467-023-44367-9](https://doi.org/10.1038/s41467-023-44367-9)

## Description
This project reproduces the following figures from the original MENDER paper:

- **Figure 5A**: MENDER-UMAP of the MERFISH aging dataset, colored by:
  - Expert annotations from the original publication
  - MENDER results

- **Figure 6B**: Cell type annotations and MENDER-predicted domains for representative TNBC patients across subtypes.

In addition to reproduction, the project explores **three novel research questions**, extending or adapting MENDER to new challenges:

1. TODO
2. TODO
3. TODO

---

## Setup Instructions

This project requires **Python 3.10.11**. You can use `pyenv` or `conda` to install the correct version.

1. Clone the repository:
    ```bash
    git clone https://github.com/ianjoshi/mender-reproduction.git
    cd mender-reproduction
    ```

2. Create and activate a virtual environment:
    ```bash
    # On Windows use
    py -3.10 -m venv mender-venv
    mender-venv/Scripts/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## References

* Yuan, Z. et al. *MENDER: Fast and scalable tissue structure identification in spatial omics data*. Nature Communications (2024).
  [DOI: 10.1038/s41467-023-44367-9](https://doi.org/10.1038/s41467-023-44367-9)

---

## Authors
- Ho Thi Ngoc Phuong \[[hothingocphuong@student.tudelft.nl](mailto:hothingocphuong@student.tudelft.nl)]
- Inaesh Joshi \[[i.joshi-2@student.tudelft.nl](mailto:i.joshi-2@student.tudelft.nl)]
- Juul Schnitzler \[[j.b.Schnitzler@student.tudelft.nl](mailto:j.b.Schnitzler@student.tudelft.nl)]