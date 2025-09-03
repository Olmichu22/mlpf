#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Para cada X en el rango, si existe el fichero de evaluación, "
                    "convierte resultados a CSV, procesa predicciones y recopila rutas.")
    parser.add_argument("start", type=int, help="Valor inicial de X (inclusive)")
    parser.add_argument("end",   type=int, help="Valor final de X (inclusive)")
    args = parser.parse_args()

    trained_models_base = "/nfs/cms/arqolmo/GPU_train/mlpf/tau_trained_models"
    sim_files_base     = "/pnfs/ciemat.es/data/cms/store/user/cepeda/FCC/FullSim/ZTauTau_SMPol_25Sept_MuonFix/"

    results = []

    for x in range(args.start, args.end + 1):
        model_dir = os.path.join(trained_models_base, f"pred_{x}")
        eval_file = os.path.join(
            model_dir,
            "showers_df_evaluation",
            "0_0_None_hdbscan_option9_v1.pt"
        )

        if not os.path.isfile(eval_file):
            print(f"[INFO] pred_{x}: fichero de evaluación no encontrado, se omite.")
            continue

        print(f"[INFO] pred_{x}: encontrado, procesando…")

        # 1) Convertir resultados a CSV
        try:
            subprocess.run(
                ["python", "mlpf_results_to_csv.py", "--path", model_dir],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] pred_{x}: fallo en mlpf_results_to_csv.py ({e.returncode}), se omite.")
            continue

        csv_path = os.path.join(model_dir, "sd_hgb.csv")

        
        if not os.path.isfile(csv_path):
            print(f"[ERROR] pred_{x}: sd_hgb.csv no generado, se omite.")
            continue
        
        # Abrimos el archivo csv
        df = pd.read_csv(csv_path)
        predicted_events = df["event_id"].unique()
        predicted_events = set(predicted_events)
        total_events = set(range(1, 1001))  # Asumiendo eventos del 1 al 1000
        missing_events = total_events - predicted_events
        # Convertimos a lista y ordenamos
        missing_events = sorted(list(missing_events))
        n_predicted = len(predicted_events)

        
        # 2) Procesar CSV para obtener fichero de predicciones
        try:
            subprocess.run(
                ["python", "predicted_data_test.py", "-i", csv_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] pred_{x}: fallo en predicted_data_test.py ({e.returncode}), se omite.")
            continue

        # 3) Determinar rutas de salida
        pred_file = csv_path.replace(".csv", "_particles.pkl")
        sim_file  = os.path.join(
            sim_files_base,
            f"out_reco_edm4hep_edm4hep_{x}.root"
        )

        results.append((x, pred_file, sim_file, n_predicted, missing_events))
        
    # 4) Guardar resultados en formato csv
    results_df = pd.DataFrame(results, columns=["id", "prediction_file", "simulation_file", "n_predicted", "missing_events"])
    output_csv = "mlpf_mapping.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"[INFO] Resultados guardados en {output_csv}")

if __name__ == "__main__":
    main()
