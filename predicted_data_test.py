import pandas as pd
import argparse
from ParticleObjects import RecoParticle
import logging
import ROOT
import pickle
import ast
import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create an argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--input_file', type=str, required=False, default="trained_models/eval_comp/sd_hgb.csv" ,help='Path to the input CSV file')
parser.add_argument('-o','--output_file', type=str, required=False, help='Path to the output pkl file')
parser.add_argument('-s', '--summary', type=bool, required=False, default=False, help='Whether to print a summary of the data')
parser.add_argument('-t', '--test', type=bool, required=False, default=False, help='Whether to run the script in test mode')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
#
args = parser.parse_args()


PID_TO_CHARGE = {
 22:0,  # Photon 
 211:1,  # Pi+
-211:-1,  # Pi-
-11:-1,  # e-
 11:1,  # e+
 13:-1,  # mu-
-13:1,  # mu+
 2112:0,  # Neutron
 321:1,  # K+
-321:-1,  # K-
}
particle_masses_4_class = {0: 0.000511, 1: 0.13957, 2: 0.939565, 3: 0.0, 4: 0.10566} # electron, CH, NH, photon, muon
pos_to_PID = {
    0: 11,  # electron
    1: 211,  # CH
    2: 2112,  # NH
    3: 22,  # photon
    4: 13   # muon
}

def process_batch(batch_df, batch, particles, logger):
    """Process a single batch (event) of the DataFrame and extract particles."""
    logger.info(f"Processing batch {batch} with {len(batch_df)} particles.")
    for _, row in batch_df.iterrows():
        pred_pos_matched = row["pred_pos_matched"]  # Only exists if is matched???
        if "nan" in pred_pos_matched:
            logger.warning(
                f"pred_pos_matched is NaN for step {batch}, skipping this particle."
            )
            continue
        # pred_energy = row['pred_showers_E']
        calibrated_energy = row["calibrated_E"]
        predicted_pos = row["pred_pid_matched"]
        predicted_pid = int(predicted_pos)  # Convert to integer
        if predicted_pid not in pos_to_PID:
            logger.warning(f"predicted_pid {predicted_pid} not in pos_to_PID, skipping this particle.")
            continue
        predicted_pid = pos_to_PID[predicted_pid]
        charge = abs(PID_TO_CHARGE.get(predicted_pid, 0))  # Default to 0 if not found
        if isinstance(pred_pos_matched, str):  # Verifica si es una cadena
          pred_pos_matched = ast.literal_eval(pred_pos_matched)  # Convierte la cadena a lista
          pred_pos_matched = np.array(pred_pos_matched)  # Asegúrate de que sea un array de numpy
          
        pred_p = pred_pos_matched / np.linalg.norm(pred_pos_matched)
        p_squared = calibrated_energy**2 - particle_masses_4_class[predicted_pos]**2
        if p_squared < 0:
          p_squared = 0  # Avoid negative square root
        
        pred_p = np.sqrt(p_squared)*pred_p  # Scale the direction by the momentum magnitude
        p4 = ROOT.TLorentzVector()
        p4.SetPxPyPzE(pred_p[0], pred_p[1], pred_p[2], calibrated_energy)
        
        particle = RecoParticle(
            p4=p4, ID=predicted_pid, charge=charge, PDGID=predicted_pid
        )
        particles[int(batch)].append(particle)

def main():
  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
  else:
    logging.getLogger().setLevel(logging.INFO)
  logger = logging.getLogger(__name__)
  logger.info("Starting the script to process the input CSV file.")
  # Read the input CSV file
  df = pd.read_csv(args.input_file)
  
  
  
  if args.summary:
    # Display the full description of all columns
    pd.set_option('display.max_columns', None)  # Ensure all columns are shown
    print(df.describe(include='all'))  # Include all columns in the description
    print(df.info())
    print(df.head())
    print(df.step.value_counts())
    print(df.event_id.unique())
    print(df.pred_pid_matched.value_counts())
    print(f"Columnas \n{df.columns}")
    exit()
  
  else:
    particles = {}
    event_numbers = df['event_id'].unique()
    for i,event in enumerate(event_numbers):
      if args.test and i >= 10:
        logger.info("Test mode enabled, processing only the first 10 events")
        break
      logger.info(f"Processing Event: {event}")
      batch_df = df[df['event_id'] == event]
      logger.debug(f"Particles in Event {event}:\n{batch_df.pid.value_counts()}")
      particles[int(event)] = []
      process_batch(batch_df, event, particles, logger)
    # Save the particles to a file
    
  if args.output_file:
    output_path = args.output_file

  else:
    output_path = args.input_file.replace('.csv', '_particles.pkl')

  with open(output_path, 'wb') as f:
    pickle.dump(particles, f)
  logger.info(f"Particles saved to {output_path}")
  logger.info("Script completed successfully.")

if __name__ == "__main__":
    main()


# output_mapping = 
# # Save the dictionary to a file
# output_mapping_path = "output_mapping.pkl"
# with open(output_mapping_path, 'wb') as f:
#     pickle.dump(output_mapping, f)

# logging.info(f"Output mapping saved to {output_mapping_path}")