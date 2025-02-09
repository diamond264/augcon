import os
import numpy as np
from fire import Fire
import scipy.io as sio


def generate(
    users=["s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    filename="_E1_A1.mat",
    data_dir="/mnt/sting/hjyoon/projects/cross/NinaproDB5/raw",
    out_dir="/mnt/sting/hjyoon/projects/cross/NinaproDB5/raw_timedomain",
):
    for user in users:
        user_filename = f"{user.upper()}{filename}"
        data_path = os.path.join(data_dir, user, user_filename)
        data = sio.loadmat(data_path)
        emg = np.array(data["emg"])
        label = np.array(data["restimulus"])
        session = np.array(data["rerepetition"])
        print(emg.shape)
        print(label.shape)

        session_dict = {}
        for i, s_ in enumerate(session):
            s = s_[0]
            if s == 0:
                continue
            if not s in session_dict:
                session_dict[s] = {"emg": [], "label": []}
            session_dict[s]["emg"].append(emg[i])
            session_dict[s]["label"].append(label[i])

        for session, session_data in session_dict.items():
            emg = np.array(session_data["emg"])
            label = np.array(session_data["label"])
            out_path = os.path.join(out_dir, user, str(session), "data.mat")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sio.savemat(out_path, {"emg": emg, "restimulus": label})


if __name__ == "__main__":
    Fire(generate)
