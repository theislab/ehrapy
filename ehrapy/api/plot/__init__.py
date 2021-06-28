import scanpy as sc


def plot_umap(ann_data):
    sc.pl.umap(ann_data, color=['Day_ICU_intime', 'age'], use_raw=False)


def plot_pca(ann_data):
    sc.pl.pca(ann_data, color=['Day_ICU_intime', 'age'], use_raw=False)
