def view_obs(ann_data):
    df = ann_data.obs
    df["Day_ICU_intime"] = ann_data.raw.X[:, :1]
    df["Service_Unit"] = ann_data.raw.X[:, 1:2]
    print(df)
