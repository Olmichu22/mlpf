import pandas as pd
import argparse
from ParticleObjects import RecoParticle
import logging
import ROOT
from ROOT import TH1F
import pickle
import os
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create an argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i','--input_file', type=str, required=False, default="trained_models/pred_particles_data.pkl" ,help='Path to the input pkl file')
parser.add_argument('-o','--output_path', type=str, required=False, help='Outputh path to save the plots')
parser.add_argument('-t', '--test', type=bool, required=False, default=False, help='Whether to run the script in test mode')
#
args = parser.parse_args()


PID_TO_CHARGE = {
 22.0:0,  # Photon 
 211.0:1,  # Pi+
-211.0:-1,  # Pi-
-11.0:-1,  # e-
 11.0:1,  # e+
 13.0:-1,  # mu-
-13.0:1,  # mu+
 2112.0:0,  # Neutron
 321.0:1,  # K+
-321.0:-1,  # K-
}

photon_momentum_hist = TH1F("photon_momentum", "Photon Momentum Distribution", 50, 0, 50)
pion_momentum_hist = TH1F("pion_momentum", "Pion Momentum Distribution", 50, 0, 50)
electron_momentum_hist = TH1F("electron_momentum", "Electron Momentum Distribution", 50, 0, 50)
muon_momentum_hist = TH1F("muon_momentum", "Muon Momentum Distribution", 50, 0, 50)
neutron_momentum_hist = TH1F("neutron_momentum", "Neutron Momentum Distribution", 50, 0, 50)
kaon_momentum_hist = TH1F("kaon_momentum", "Kaon Momentum Distribution", 50, 0, 50)

histograms = {
    22.0: photon_momentum_hist,
    211.0: pion_momentum_hist,
   -11.0: electron_momentum_hist,
   -13.0: muon_momentum_hist,
   2112.0: neutron_momentum_hist,
    321.0: kaon_momentum_hist
}

def plot_and_save_histograms(histograms, output_path, logger):
    """Plot and save histograms to the specified output path."""
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    for pid, hist in histograms.items():
        canvas = ROOT.TCanvas(f"canvas_{pid}", f"Histogram for PID {pid}", 800, 600)
        title = f"Momentum Distribution for PID {pid}"
        hist.SetTitle(title)
        hist.SetXTitle("Momentum (GeV/c)")
        hist.SetYTitle("Counts")
        hist.Draw()
        canvas.SaveAs(os.path.join(output_path, f"histogram_{pid}.png"))
        logger.info(f"Saved histogram for PID {pid} to {output_path}.")

def process_event(event, event_info, histograms, logger):
    """Process an event and extract histograms information."""
    logger.info(f"Processing event {event} with {len(event_info)} particles.")
    for recoparticle in event_info:
      pid = recoparticle.getPDG()
      if pid in histograms:
        hist = histograms[pid]
        momentum = recoparticle.getMomentum().P()
        hist.Fill(momentum)
        logger.info(f"Filled histogram for PID {pid} with momentum {momentum}.")
      else:
        logger.warning(f"PID {pid} not found in histograms, skipping.")

def main():
  logger = logging.getLogger(__name__)
  logger.info("Starting the script to process the input CSV file.")
  input_file = args.input_file
  # Cargamos el archivo pkl
  with open(input_file, 'rb') as f:
    events_data = pickle.load(f)
  
  for event in events_data.keys():
    logger.info(f"Processing event: {event}")
    event_info = events_data[event]
    process_event(event, event_info, histograms, logger)
    
  if args.output_path:
    output_path = args.output_path if args.output_path else "output_histograms"
    plot_and_save_histograms(histograms, output_path, logger)
  else:
    logger.error("No output path specified. Please provide a valid output path using --output_path argument.")
    
    
if __name__ == "__main__":
    main()