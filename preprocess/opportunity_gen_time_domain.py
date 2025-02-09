import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire


def generate(
    users=["S1", "S2", "S3", "S4"],
    data_dir="/mnt/sting/hjyoon/projects/cross/Opportunity/OpportunityUCIDataset/dataset",
    out_dir="/mnt/sting/hjyoon/projects/cross/Opportunity/csvs/",
):
    column_name_file = os.path.join(data_dir, "column_names.txt")
    label_legend_file = os.path.join(data_dir, "label_legend.txt")

    # generate list of columns
    with open(column_name_file, "r") as f:
        column_names_ = f.read().splitlines()
    column_names = []
    for column_name_ in column_names_:
        if column_name_.startswith("Column: "):
            column_name = " ".join(column_name_.split(" ")[2:])
            column_name = column_name.split(";")[0]
            column_names.append(column_name)

    # generate map of labels
    with open(label_legend_file, "r") as f:
        label_map_lines = f.read().splitlines()
    label_map = {}
    for label_map_line in label_map_lines:
        elems = label_map_line.split("-")
        if len(elems) == 3:
            # check if elems[0] is a number
            if elems[0].strip().isdigit():
                label_map[int(elems[0].strip())] = elems[2].strip()

    # target_columns = []
    target_column_idcs = []
    for imu_channel in ["accX", "accY", "accZ"]:
        target_column = f"Accelerometer RWR {imu_channel}"
        # target_columns.append(f"RWR {imu_channel}")
        target_column_idcs.append(column_names.index(target_column))
    target_column_idcs.append(column_names.index("Locomotion"))

    sessions = [1, 2, 3, 4, 5]
    for user in users:
        concat_data = pd.DataFrame()
        for session in tqdm(sessions):
            user_filename = f"{user}-ADL{session}.dat"
            data_path = os.path.join(data_dir, user_filename)
            data = pd.read_csv(data_path, sep=" ", header=None)
            data = data.iloc[:, target_column_idcs]
            # drop NaNs
            data = data.dropna()
            # drop rows with label 0
            data = data[data.iloc[:, -1] != 0]
            # map labels
            data.iloc[:, -1] = data.iloc[:, -1].map(label_map)
            # add one more column for session
            data["domain"] = session
            # change column names
            data.columns = ["accx", "accy", "accz", "gt", "domain"]
            # change order of gt and domain
            data = data[["accx", "accy", "accz", "domain", "gt"]]
            concat_data = pd.concat([concat_data, data])
        # standardize data from -1 to 1 (min-max scaling for the first 3 columns)
        concat_data.iloc[:, :-2] = (
            concat_data.iloc[:, :-2] - concat_data.iloc[:, :-2].min()
        ) / (concat_data.iloc[:, :-2].max() - concat_data.iloc[:, :-1].min())
        concat_data.iloc[:, :-2] = concat_data.iloc[:, :-2] * 2 - 1

        out_path = os.path.join(out_dir, f"{user}_timedomain.csv")
        concat_data.to_csv(out_path, index=False)


if __name__ == "__main__":
    Fire(generate)
