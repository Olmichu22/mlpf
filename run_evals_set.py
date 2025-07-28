#!/usr/bin/env python3
import subprocess
import sys
import pandas as pd
import time
def main():
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <inicio> <fin>")
        print("Ejemplo: python run_evals.py 10 20")
        sys.exit(1)

    inicio, fin = map(int, sys.argv[1:])
    errors = {}
    for i in range(inicio, fin + 1):
        data_test = f"/nfs/cms/arqolmo/GPU_train/mlpf/test_trees/test_tree_{i}.root"
        model_prefix = f"/nfs/cms/arqolmo/GPU_train/mlpf/trained_models/pred_{i}/"

        cmd = [
            "python", "-m", "src.train_lightning1",
            "--data-test",      data_test,
            "--data-config",    "config_files/config_hits_track_v4.yaml",
            "-clust", "-clust_dim", "3",
            "--network-config", "src/models/wrapper/example_mode_gatr_e.py",
            "--model-prefix",   model_prefix,
            "--wandb-displayname", "eval_gun_drlog",
            "--num-workers",    "0",
            "--gpus",           "1",
            "--batch-size",     "10",
            "--start-lr",       "1e-3",
            "--num-epochs",     "100",
            "--optimizer",      "ranger",
            "--fetch-step",     "0.1",
            "--condensation",
            "--log-wandb",
            "--wandb-projectname", "mlpf_debug_eval",
            "--frac_cluster_loss", "0",
            "--qmin",           "1",
            "--use-average-cc-pos", "0.99",
            "--lr-scheduler",   "reduceplateau",
            "--tracks",
            "--correction",
            "--ec-model",       "gatr-neutrals",
            "--regress-pos",
            "--add-track-chis",
            "--load-model-weights",
                "/nfs/cms/arqolmo/GPU_train/mlpf/trained_models/"
                "E_PID_02122024_dr05_s6500_3layer_pid_GTClusters_"
                "all_classes_PID_epoch0step4500.ckpt",
            "--freeze-clustering",
            "--predict",
            "--regress-unit-p",
            "--PID-4-class",
            "--n-layers-PID-head", "3",
            "--separate-PID-GATr"
        ]

        print(f"\n=== Ejecutando iteración {i}: ===")
        print(" ".join(cmd))
        try:
            # check=True lanza CalledProcessError si exit code != 0
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Iteración {i} falló con código {e.returncode}. Continuando con la siguiente.\n")
            errors[i] = str(e)
            continue

    errors_df = pd.DataFrame.from_dict(errors, orient='index', columns=['Error'])
    if not errors_df.empty:
        time_now = time.strftime("%Y%m%d-%H%M%S")
        errors_df.to_csv(f'errors_{time_now}.csv', index_label='Iteration')
        print("\nErrores registrados en 'errors.csv'.")
if __name__ == "__main__":
    main()
