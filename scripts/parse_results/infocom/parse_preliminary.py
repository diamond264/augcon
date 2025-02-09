from glob import glob
import pandas as pd
import fire


def run(path, setting="", shot=""):
    df_list = []
    shot_paths = glob(f"{path}/{shot}*")
    for shot_path in shot_paths:
        shot = shot_path.split("/")[-1]
        setting_paths = glob(f"{shot_path}/{setting}*")
        for setting_path in setting_paths:
            setting = setting_path.split("/")[-1]
            seed_paths = glob(f"{setting_path}/*")
            for seed_path in seed_paths:
                seed = seed_path.split("/")[-1]
                log_files = glob(f"{seed_path}/*.log")
                f1scores = []
                for log_file in log_files:
                    user = log_file.split("/")[-1].split(".")[0]
                    with open(log_file, "r", encoding="utf-8") as f:
                        last_line = f.read().strip().split("\n")[-1]
                        f1score = float(
                            last_line.strip().split("F1: ")[1].strip().split(",")[0]
                        )
                        df_list.append(
                            {
                                "setting": setting,
                                "shot": shot,
                                "user": user,
                                "seed": seed,
                                "f1score": f1score,
                            }
                        )

    df = pd.DataFrame(df_list)
    # average by different seeds
    df = df.groupby(["setting", "shot", "user", "seed"])["f1score"].mean().reset_index()
    df_ = df["f1score"].mean()
    print(df_)
    df_ = (
        df.groupby(["user", "shot", "setting"])["f1score"]
        .std()
        .groupby(["shot", "setting"])
        .mean()
    )
    df_ = df_.reindex(["1shot", "2shot", "5shot", "10shot"], level=0)
    print(df_)


if __name__ == "__main__":
    fire.Fire(run)
