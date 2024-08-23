from glob import glob
import fire


def run(path, setting_="linear"):
    shot_paths = glob(f"{path}/*")
    for shot_path in shot_paths:
        setting_paths = glob(f"{shot_path}/{setting_}")
        for setting_path in setting_paths:
            log_files = glob(f"{setting_path}/*/*.log")
            f1scores = []
            for log_file in log_files:
                with open(log_file, "r", encoding="utf-8") as f:
                    last_line = f.read().strip().split("\n")[-1]
                    f1score = float(
                        last_line.strip().split("F1: ")[1].strip().split(",")[0]
                    )
                    f1scores.append(float(f1score))
            avg_f1score = sum(f1scores) / len(f1scores)
            shot = shot_path.split("/")[-1]
            setting = setting_path.split("/")[-1]
            print(f"[{setting}-shot{shot}] Average F1 score: {avg_f1score}")


if __name__ == "__main__":
    fire.Fire(run)
