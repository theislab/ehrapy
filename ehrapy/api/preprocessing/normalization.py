from typing import Optional

from anndata import AnnData


class Normalization:
    """Provides functions to normalize continuous features"""

    @staticmethod
    def identity(adata: AnnData, copy: bool = False) -> Optional[AnnData]:
        """Returns the original object without any normalisation

        Created as a template during development. Should be removed before merging.

        Args:
            adata: :class:`~anndata.AnnData` object containing X to normalize values in
            copy: Whether to return a copy with the normalized data.

        Returns:
            :class:`~anndata.AnnData` object with normalized X
        """
        adata_to_act_on = adata
        if copy:
            adata_copy = adata.copy()
            adata_to_act_on = adata_copy

        return adata_to_act_on
