import scanpy as sc


def umap(ann_data):
    sc.pp.neighbors(ann_data)
    sc.tl.umap(ann_data)
    sc.pl.umap(ann_data, color=['Day_ICU_intime', 'age'], use_raw=False)


def pca(ann_data):
    sc.tl.pca(ann_data, svd_solver='arpack')
    sc.pl.pca(ann_data, color=['Day_ICU_intime', 'age'], use_raw=False)
