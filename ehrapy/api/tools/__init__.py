import scanpy as sc


def calc_pca(ann_data):
    sc.tl.pca(ann_data, svd_solver='arpack')


def calc_umap(ann_data):
    sc.pp.neighbors(ann_data)
    sc.tl.umap(ann_data)
