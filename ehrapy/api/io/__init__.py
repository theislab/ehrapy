import numpy as np
from anndata import AnnData
import pandas as pd


def init_anndata_from_df(path):
    df = pd.read_csv(path)
    df.fillna(df.mean(), inplace=True)
    df.insert(0, 'day_icu_intimee', df.pop('day_icu_intime'))
    df.insert(1, 'service_unitt', df.pop('service_unit'))

    df['day_icu_intimee'] = df['day_icu_intimee'].str.strip()
    df.insert(0, 'Patient_id', df[['age', 'hospital_los_day', 'iv_day_1']].sum(axis=1).map(hash))
    df = df.astype({"Patient_id": str})
    test = np.array(df.pop('Patient_id'))

    df2 = df.copy()
    # one hot encode categoricals and store it in obsm
    ml_encoded_day_icu_intimee = pd.get_dummies(df2['day_icu_intimee'], prefix='day_icu_intimee', prefix_sep='_')
    ml_encoded_service_unitt = pd.get_dummies(df2['service_unitt'], prefix='service_unitt', prefix_sep='_')

    df2.insert(1, 'day_icu_intimee_Friday', ml_encoded_day_icu_intimee['day_icu_intimee_Friday'])
    df2.insert(2, 'day_icu_intimee_Saturday', ml_encoded_day_icu_intimee['day_icu_intimee_Saturday'])
    df2.insert(3, 'day_icu_intimee_Sunday', ml_encoded_day_icu_intimee['day_icu_intimee_Sunday'])
    df2.insert(4, 'day_icu_intimee_Monday', ml_encoded_day_icu_intimee['day_icu_intimee_Monday'])
    df2.insert(5, 'day_icu_intimee_Tuesday', ml_encoded_day_icu_intimee['day_icu_intimee_Tuesday'])
    df2.insert(6, 'day_icu_intimee_Wednesday', ml_encoded_day_icu_intimee['day_icu_intimee_Wednesday'])
    df2.insert(7, 'day_icu_intimee_Thursday', ml_encoded_day_icu_intimee['day_icu_intimee_Thursday'])
    df2.pop('day_icu_intimee')
    df2.pop('service_unitt')

    df2.insert(8, 'service_unitt_FICU', ml_encoded_service_unitt['service_unitt_FICU'])
    df2.insert(9, 'service_unitt_MICU', ml_encoded_service_unitt['service_unitt_MICU'])
    df2.insert(10, 'service_unitt_SICU', ml_encoded_service_unitt['service_unitt_SICU'])

    ann_data_raw = AnnData(np.array(df), dtype="object", obs=dict(obs_names=test), var=dict(var_names=np.array(df.columns)))
    ann_data = AnnData(np.array(df2), obs=dict(obs_names=test), var=dict(var_names=np.array(df2.columns)))
    # set raw to ann_data obj with label encoded X
    # we need a copy here
    ann_data.raw = ann_data_raw

    cleanup_nums = {"day_icu_intimee": {"Friday": 5, "Saturday": 6, "Sunday": 7, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
                                        "Thursday": 4},
                    "service_unitt": {"SICU": 1, "MICU": 2, "FICU": 3}}

    # store label mappings in uns
    ann_data.uns['label_mapping'] = cleanup_nums

    return ann_data


def encode_vars(ann_data, one_hot_encode, label_encode = []):
    pass
