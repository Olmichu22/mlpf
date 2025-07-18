#!/bin/bash
# ------------------------------------------------------------
#  make_pf_tree.sh
#
#  Convierte un archivo out_reco_edm4hep_REC.edm4hep.root
#  en un árbol de entrenamiento pf_tree_<seed>.root
#
#  ARGUMENTOS:
#    1) HOMEDIR     -> directorio base del repo mlpf  (contiene condor/, guns/, etc.)
#    2) RECO_FILE   -> ruta al out_reco_edm4hep_REC.edm4hep.root
#    3) SEED        -> identificador numérico único (ej. ClusterIdProcId de Condor)
#    4) OUTPUTDIR   -> directorio final donde copiar el pf_tree
#
#  EJEMPLO:
#    ./make_pf_tree.sh $PWD /eos/.../out_reco_edm4hep_REC.edm4hep.root 12345 /eos/user/.../pf_trees
# ------------------------------------------------------------

set -euo pipefail

HOMEDIR="$1"
RECO_FILE="$2"
SEED="$3"
OUTPUTDIR="$4"

# --- Directorio de trabajo temporal (cada job en subcarpeta distinta) ---
WORKDIR="${PWD}/job_${SEED}"
mkdir -p "${WORKDIR}"
cd       "${WORKDIR}"

# --- Entorno Key4HEP ---
source /cvmfs/sw.hsf.org/key4hep/setup.sh

# --- Copiamos los dos scripts auxiliares que viven en el repo -----------
cp "${HOMEDIR}/condor/make_pftree_clic_bindings.py" .
cp "${HOMEDIR}/condor/tree_tools.py"                 .

# --- Ejecución ----------------------------------------------------------
python make_pftree_clic_bindings.py \
  "${RECO_FILE}"            \   # entrada
  "pf_tree_${SEED}.root"    \   # salida
  False                     \   # sin evaluar clustering
  False                         # sin guardar hits sueltos

# --- Copiado a EOS/xrootd/afs ------------------------------------------
mkdir -p "${OUTPUTDIR}"
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py \
       "pf_tree_${SEED}.root" \
       "${OUTPUTDIR}/pf_tree_${SEED}.root"

echo "✓  pf_tree_${SEED}.root escrito en ${OUTPUTDIR}"
