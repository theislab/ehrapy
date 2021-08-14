from typing import Dict

from anndata import AnnData

from ehrapy.api.encode.encode import Encoder


def encode(ann_data: AnnData, autodetect: bool = False, categoricals_encode_mode: Dict = None) -> AnnData:
    return Encoder.encode(ann_data, autodetect, categoricals_encode_mode)
