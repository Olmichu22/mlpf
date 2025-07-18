import pandas as pd
import numpy as np
from src.utils.pid_conversion import pid_conversion_dict
import  argparse
import os


def open_mlpf_dataframe(path_mlpf, neutrals_only=False):
    """
    Opens and processes a dataframe containing MLPF (Machine Learning Particle Flow) data.

    Args:
      path_mlpf (str): Path to the pickle file containing the MLPF data.
      neutrals_only (bool, optional): If True, filters the dataframe to include only neutral particles 
        (pid values 130, 2112, and 211). Defaults to False.

    Returns:
      tuple: A tuple containing:
        - sd (pd.DataFrame): The processed dataframe with additional columns and filtering applied.
        - matched (pd.DataFrame): A subset of the dataframe where both `pred_showers_E` and 
          `reco_showers_E` are not NaN.

    Raises:
      ValueError: If the dataframe contains particle IDs (`pid`) that are not present in 
        `pid_conversion_dict` and are not NaN.

    Notes:
      - The function maps particle IDs (`pid`) to a new classification (`pid_4_class_true`) using 
        `pid_conversion_dict`.
      - If the column `pred_pid_matched` exists, values less than -1 are replaced with NaN.
      - Prints a warning message for any `pid` not found in `pid_conversion_dict` before raising an error.
    """
    data = pd.read_pickle(path_mlpf)
    if neutrals_only:
        sd = pd.concat(
            [
                data[data["pid"] == 130],
                data[data["pid"] == 2112],
                data[data["pid"] == 211],
            ]
        )
    else:
        sd = data
    mask = (~np.isnan(sd["pred_showers_E"])) * (~np.isnan(sd["reco_showers_E"]))
    sd["pid_4_class_true"] = sd["pid"].map(pid_conversion_dict)
    for item in sd.pid.unique():
        if item not in pid_conversion_dict.keys() and not pd.isna(item):
            print(f"Item {item} not in pid_conversion_dict")
            raise ValueError
    if "pred_pid_matched" in sd.columns:
        sd.loc[sd["pred_pid_matched"] < -1, "pred_pid_matched"] = np.nan
    matched = sd[mask]
    return sd, matched
  
  
  
def main():
  parser = argparse.ArgumentParser(description="Convert MLPF results to CSV format for further analysis.")
  parser.add_argument("--path", type=str,
                  help="Path to the folder with the training in which checkpoints are saved",
                  required=True)
  args = parser.parse_args()
  path = args.path
  input_path = os.path.join(path, "showers_df_evaluation/0_0_None_hdbscan_option9_v1.pt")
  output_path = os.path.join(path, "sd_hgb.csv")
  sd_hgb, _ = open_mlpf_dataframe(input_path, False)
  sd_hgb.to_csv(output_path, index=False)
  
  
  
  
if __name__ == "__main__":
  main()