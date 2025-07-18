#!/usr/bin/env python
# filepath: visualize_hits_clusters.py
import uproot
import torch
import numpy as np
import plotly.graph_objects as go
import os
import argparse
from pathlib import Path
from plotly.io import to_html

global pos_to_PID, pid_map, real_pid_map, real_particle_colors, particle_colors
# Mapeo de PID a nombres de partículas
pos_to_PID = {
    0: 11,    # electron
    1: 211,   # CH (Hadrón cargado)
    2: 2112,  # NH (Hadrón neutro)
    3: 22,    # photon
    4: 13     # muon
}
# Mapeo para nombres amigables
pid_map = {
    11: "Electron",
    211: "Charged Hadron",
    2112: "Neutral Hadron",
    22: "Photon",
    13: "Muon"
}
real_pid_map = {
    11: "Electron",
    -11: "Positron",
    12: "Electron Neutrino",
    -12: "Electron Antineutrino",
    211: "Charged Pion (-)",
    -211: "Charged Pion (+)",
    111: "Neutral Pion",
    2112: "Neutron",
    22: "Photon",
    13: "Muon",
    -13: "Antimuon",
    14: "Muon Neutrino",
    -14: "Muon Antineutrino",
    16: "Tau Neutrino",
    -16: "Tau Antineutrino",
    130: "Neutral Kaon",
    321: "Charged Kaon (+)",
    -321: "Charged Kaon (-)",
    310: "Neutral Kaon (L)",
    311: "Neutral Kaon (S)",
    221: "Eta",
    223: "Omega",
    -1: "Unknown"
}
pid_to_mass = {
    11: 0.000511,
    -11: 0.000511,
    12: 0,
    -12: 0,
    211: 0.13957,
    -211: 0.13957,
    111: 0.135,
    2112: 0.939565,
    22: 0,
    13: 0.10566,
    -13: 0.10566,
    14: 0,
    -14: 0,
    16: 0,
    -16: 0,
    130: 0.497611,
    321: 0.493677,
    -321: 0.493677,
    310: 0.497611,
    311: 0.497611,
    221: 0.547862,
    223: 0.78266
}
# pid_to_mass = {0: 0.000511, 1: 0.13957, 2: 0.939565, 3: 0.0, 4: 0.10566} # electron, CH, NH, photon, muon


real_particle_colors = {
    "Electron":"cyan",
    "Positron":"cyan",
    "Electron Neutrino":"lightblue",
    "Electron Antineutrino":"lightblue",
    "Charged Pion (-)":"magenta",
    "Charged Pion (+)":"magenta",
    "Neutral Pion": "purple",
    "Neutron":"purple",
    "Photon": "orange",
    "Muon": "lime",
    "Antimuon": "lime",
    "Muon Neutrino": "lightgreen",
    "Muon Antineutrino":"lightgreen",
    "Tau Neutrino": "lightred",
    "Tau Antineutrino": "lightred",
    "Neutral Kaon":"purple",
    "Charged Kaon (+)":"magenta",
    "Charged Kaon (-)":"magenta",
    "Neutral Kaon (L)":"purple",
    "Neutral Kaon (S)":"purple",
    "Eta": "purple",
    "Omega": "purple",
    "Unknown": "gray"
}
# Colores específicos por tipo de partícula
particle_colors = {
    "Electron": "cyan",
    "Charged Hadron": "magenta", 
    "Neutral Hadron": "purple",
    "Photon": "orange",
    "Muon": "lime",
    "Unknown": "gray"
}

def load_tree_data(tree_file, tree_name="events"):
    """Carga los hits desde el archivo ROOT original"""
    with uproot.open(tree_file) as f:
        # Listar las claves disponibles para diagnóstico
        print(f"Claves disponibles en el archivo: {f.keys()}")
        
        tree = f[tree_name]
        
        # Listar las ramas disponibles para diagnóstico
        print(f"Ramas disponibles en el árbol: {tree.keys()}")
        
        # Extraer posiciones de hits (comprobando primero si existen las ramas)
        try:
            hit_x = tree["hit_x"].array(library="np")
            hit_y = tree["hit_y"].array(library="np") 
            hit_z = tree["hit_z"].array(library="np")
            hit_energy = tree["hit_e"].array(library="np")
            hit_type = tree["hit_type"].array(library="np")
            hits_part_pid = tree["part_pid"].array(library="np")
            # Cargar la relación hit-partícula (esta es la clave)
            # hit_genlink = tree["hit_genlink"].array(library="np")
            
            # return hit_x, hit_y, hit_z, hit_energy, hit_type, hit_genlink
            # hit_particle_id = tree["hit_genlink"].array(library="np")
                        # Si necesitas considerar múltiples contribuciones
            weights0 = tree["hit_genweight0"].array(library="np")
            weights1 = tree["hit_genweight1"].array(library="np")
            weights2 = tree["hit_genweight2"].array(library="np")
            weights3 = tree["hit_genweight3"].array(library="np")
            weights4 = tree["hit_genweight4"].array(library="np")
            links0   = tree["hit_genlink0"].array(library="np")
            links1   = tree["hit_genlink1"].array(library="np")
            links2   = tree["hit_genlink2"].array(library="np")
            links3   = tree["hit_genlink3"].array(library="np")
            links4   = tree["hit_genlink4"].array(library="np")
            
            hit_to_particle_by_event = []

            for w0, w1, w2, w3, w4, l0, l1, l2, l3, l4 in zip(weights0,
                                      weights1,
                                      weights2,
                                      weights3,
                                      weights4,  
                                      links0,
                                      links1,
                                      links2,
                                      links3,
                                      links4):
                
                weights_ev = np.stack([w0, w1, w2, w3, w4])  # shape (2, n_hits_ev)
                links_ev   = np.stack([l0, l1, l2, l3, l4])  # shape (2, n_hits_ev)
                max_idx    = np.argmax(weights_ev, axis=0)
                best_links = links_ev[max_idx, np.arange(len(max_idx))]
                hit_to_particle_by_event.append(best_links.tolist())
            # print(hit_to_particle_by_event[0])
            return hit_x, hit_y, hit_z, hit_energy, hit_type, hit_to_particle_by_event, hits_part_pid
            
        except KeyError as e:
            print(f"Error: No se encontró la rama {e}.")
            print("Por favor, verifica los nombres de las ramas en el árbol.")
            
            # Sugerencia: buscar nombres similares
            possible_x = [key for key in tree.keys() if 'x' in key.lower()]
            possible_energy = [key for key in tree.keys() if 'energy' in key.lower() or 'e' == key.lower()]
            print(f"Posibles ramas para posición x: {possible_x}")
            print(f"Posibles ramas para energía: {possible_energy}")
            
            raise

def load_model_predictions(pred_file):
    """Carga las partículas reconstruidas por el modelo"""
    try:
        # Cargar usando pickle (ya que el archivo es un DataFrame)
        import pickle
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Predicciones cargadas con éxito: {type(data)}")
        print(f"Columnas disponibles: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error al cargar predicciones: {e}")
        raise
    
# def load_model_predictions(pred_file):
#     """Carga las partículas reconstruidas por el modelo desde un archivo .pt"""
#     try:
#         # Cargar usando torch
#         data = torch.load(pred_file)
#         print(f"Predicciones cargadas con éxito: {type(data)}")
#         if isinstance(data, dict):
#             print(f"Claves disponibles: {list(data.keys())}")
#         return data
#     except Exception as e:
#         print(f"Error al cargar predicciones: {e}")
#         raise

def add_tracks_from_hits(fig, hit_positions, hit_type, hit_particle_id,hits_part_pid, event_idx):
    """Añade tracks conectando hits de tracking de la misma partícula"""
    
    # Filtrar solo hits de tracking (tipo 2)
    track_mask = hit_type[event_idx] == 2
    
    if np.sum(track_mask) == 0:
        return fig  # No hay hits de tracking
        
    track_x = hit_positions[0][event_idx][track_mask]
    track_y = hit_positions[1][event_idx][track_mask]
    track_z = hit_positions[2][event_idx][track_mask]
    track_particle_ids = hit_particle_id[event_idx]
    track_particle_ids = np.array(track_particle_ids)[track_mask]
    # print(set(track_particle_ids))
    # Para cada partícula única, conectar sus hits como una línea
    # print(track_x.shape)
    # print(track_particle_ids.shape)
    part_pid = hits_part_pid[event_idx]
    for particle_id in np.unique(track_particle_ids):
        if particle_id == 0:  # Ignorar ruido/hits no asignados
            continue
        # Filtrar hits solo de esta partícula
        particle_mask = track_particle_ids == particle_id
        
        if np.sum(particle_mask) < 2:
            continue  # Necesitamos al menos dos puntos para una línea
        print("Dibujando track para partícula:", particle_id)
        
        # Ordenar hits por posición z (profundidad)
        p_x = track_x[particle_mask]
        p_y = track_y[particle_mask]
        p_z = track_z[particle_mask]
        
        # Ordenar por coordenada Z
        sort_idx = np.argsort(p_z)
        p_x = p_x[sort_idx]
        p_y = p_y[sort_idx]
        p_z = p_z[sort_idx]
        
        # Añadir línea a la visualización
        fig.add_trace(go.Scatter3d(
            x=p_x, y=p_y, z=p_z,
            mode='lines',
            line=dict(color='yellow', width=4),
            name=f'Track {part_pid[int(particle_id)]}',
            hoverinfo='name'
        ))
    
    return fig

def plot_hits(fig, hit_positions, hit_energy, hit_type, hit_id, hits_part_pid, event_idx):
    
    # Filtrar por evento
    x_hits = hit_positions[0][event_idx]
    y_hits = hit_positions[1][event_idx]
    z_hits = hit_positions[2][event_idx]
    e_hits = hit_energy[event_idx]
    types = hit_type[event_idx]

    
    # Paleta de colores para tipos de hits
    color_map = {0: 'blue',    # ECAL hits
                 1: 'red',     # HCAL hits 
                 2: 'green'}   # Tracking hits
    
    # Añadir hits al gráfico con tamaño aumentado
    colors = [color_map.get(t, 'gray') for t in types]
    sizes = np.clip(np.log10(e_hits+1)*100, 3, 12)  # Tamaños aumentados
    
    # Trazar los hits
    fig.add_trace(go.Scatter3d(
        x=x_hits, y=y_hits, z=z_hits,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8
        ),
        name='Hits',
        hovertext=[f"Hit: E={e:.3f}, Tipo={t}" for e, t in zip(e_hits, types)]
    ))
    
    return fig

def get_particle_momentum(pos, energy, pid):

    momentum = pos / np.linalg.norm(pos)
    p_squared = energy**2 - pid_to_mass[pid]**2
    if p_squared < 0:
        p_squared = 0  # Avoid negative square root

    momentum = np.sqrt(p_squared)*momentum  # Scale the direction by the momentum magnitude
    return momentum

def plot_reco_particles(fig, pred_particle_positions, pred_particle_energy, pred_particle_types=None, event_idx=0):
    
    if event_idx in pred_particle_positions:
        p_energy = pred_particle_energy[event_idx]
        p_types = pred_particle_types[event_idx] if pred_particle_types and event_idx in pred_particle_types else ["Unknown"] * len(p_energy)
        
        # Convertir el tipo numérico al PID correspondiente
        particle_ids = [pos_to_PID.get(int(t), -1) if not np.isnan(t) else -1 for t in p_types]
        
        # Color según tipo de partícula
        colors = [particle_colors.get(pid_map.get(pid, "Unknown"), "gray") for pid in particle_ids]
        # Origen y direcciones
        origins = np.array([0, 0, 0])
        dirs    = pred_particle_positions[event_idx]
        origins = np.tile(origins, (len(dirs), 1))

        
        # Position Plot
        # Preparamos arrays intercalando None para que Plotly dibuje líneas separadas
        x_lines, y_lines, z_lines = [], [], []
        for o, d in zip(origins, dirs):
            tip = o + d
            # Cada segmento va: origen → punta → (None)
            x_lines += [o[0], tip[0], None]
            y_lines += [o[1], tip[1], None]
            z_lines += [o[2], tip[2], None]
        
        hover_info = [f"Reco Particle: E={e:.3f}GeV, Type={pid_map.get(pid, f'Unknown')} (PID:{pid})" 
                      for e, pid in zip(p_energy, particle_ids)]
        # Expandir la información hover para cada segmento
        expanded_hovertext = []
        for info in hover_info:
            # Para cada vector, añadir el mismo texto para origen y punta, y texto vacío para None
            expanded_hovertext += [info, info, ""]
            
        colors_lines = []
        for c in colors:
            colors_lines += [c, c, 'rgba(0,0,0,0)']
        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            name='Reco Position Vector',
            line=dict(
            color=colors_lines,   # color CSS, nombre o hex
            width=4,
            dash='dot'
        ),
            hovertext=expanded_hovertext
        ))

        # Final position Points
        # Marcadores en las puntas
        tips = origins + dirs  # origen + vector = punta
        magnitudes = np.linalg.norm(dirs, axis=1)

        fig.add_trace(go.Scatter3d(
            x=tips[:, 0], y=tips[:, 1], z=tips[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,      
            ),
            name='Reco Particle',
            hovertext=[f"Reco Particle: E={e:.3f}GeV, Type={pid_map.get(pid, f'Unknown')} (PID:{pid})" 
                      for e, pid in zip(p_energy, particle_ids)]
        ))
        
        # Momentum plot
        particles_p = []
        for idx, reco_particle_pid in enumerate(particle_ids):
            momentum = get_particle_momentum(dirs[idx], p_energy[idx], reco_particle_pid)
            particles_p.append(momentum*1000)
        
        particles_p = np.array(particles_p)
        # Añadir vectores de momento
        x_lines_p, y_lines_p, z_lines_p = [], [], []
        for o, d in zip(tips, particles_p):
            tip = o + d
            # Cada segmento va: origen → punta → (None)
            x_lines_p += [o[0], tip[0], None]
            y_lines_p += [o[1], tip[1], None]
            z_lines_p += [o[2], tip[2], None]
        
        colors_lines_p = []
        for c in colors:
            colors_lines_p += [c, c, 'rgba(0,0,0,0)']
        
        hover_info = [
                f"Momentum: {np.linalg.norm(p):.3f} MeV/c, "
                f"Type={pid_map.get(pid, 'Unknown')} (PID:{pid})"
                for p, pid in zip(particles_p, particle_ids)
            ]
        # Expandir la información hover para cada segmento
        expanded_hovertext = []
        for info in hover_info:
            # Para cada vector, añadir el mismo texto para origen y punta, y texto vacío para None
            expanded_hovertext += [info, info, ""]    
            
        fig.add_trace(go.Scatter3d(
            x=x_lines_p, y=y_lines_p, z=z_lines_p,
            mode='lines',
            name='Reco Momentum Vectors',
            line=dict(
                color=colors_lines_p,   # color CSS, nombre o hex
                width=4,
            ),
            hovertext=expanded_hovertext
        ))
        
    
    return fig

def plot_gen_particle(fig, real_particle_positions, real_particle_energy, real_particle_types=None, event_idx=0):
    
    if event_idx in real_particle_positions:
        r_pos = real_particle_positions[event_idx] 
        r_energy = real_particle_energy[event_idx]
        # print(real_particle_types[event_idx])
        r_types = real_particle_types[event_idx] if real_particle_types and event_idx in real_particle_types else ["Unknown"] * len(r_energy)
        
        # Convertir el tipo numérico al PID correspondiente
        # print(r_types)
        particle_ids = [t if not np.isnan(t) else -1 for t in r_types]
        
        # Color según tipo de partícula
        colors = [real_particle_colors.get(real_pid_map.get(pid, "Unknown"), "gray") for pid in particle_ids]
        
        fig.add_trace(go.Scatter3d(
            x=r_pos[:, 0], y=r_pos[:, 1], z=r_pos[:, 2],
            mode='markers',
            marker=dict(
                size=14,
                color=colors,
                symbol='cross',
                line=dict(color='black', width=1.5)
            ),
            name='Gen Particle',
            hovertext=[f"Partícula: E={e:.3f}, Tipo={real_pid_map.get(pid, f'Unknown')} (PID:{pid})" 
                      for e, pid in zip(r_energy, particle_ids)]
        ))
        
        particles_p = []
        for idx, gen_particle_pid in enumerate(particle_ids):
            momentum = get_particle_momentum(r_pos[idx], r_energy[idx], gen_particle_pid)
            particles_p.append(momentum*1000)
        
        particles_p = np.array(particles_p)
        # Añadir vectores de momento
        x_lines_p, y_lines_p, z_lines_p = [], [], []
        for o, d in zip(r_pos, particles_p):
            tip = o + d
            # Cada segmento va: origen → punta → (None)
            x_lines_p += [o[0], tip[0], None]
            y_lines_p += [o[1], tip[1], None]
            z_lines_p += [o[2], tip[2], None]
        
        hover_info = [
            f"Momentum: {np.linalg.norm(p):.3f} MeV/c, "
            f"Type={real_pid_map.get(pid, 'Unknown')} (PID:{pid})"
            for p, pid in zip(particles_p, particle_ids)
        ]
        # Expandir la información hover para cada segmento
        expanded_hovertext = []
        for info in hover_info:
            # Para cada vector, añadir el mismo texto para origen y punta, y texto vacío para None
            expanded_hovertext += [info, info, ""]
        
        colors_lines_p = []
        for c in colors:
            colors_lines_p += [c, c, 'rgba(0,0,0,0)']
        fig.add_trace(go.Scatter3d(
            x=x_lines_p, y=y_lines_p, z=z_lines_p,
            mode='lines',
            name='Gen Momentum Vectors',
            line=dict(
                color=colors_lines_p,   # color CSS, nombre o hex
                width=4,
                dash='dash'
            ),
            hovertext=expanded_hovertext
        ))
        
    return fig


def save_html_with_summary(fig, summary_html, output_path):
    """Guarda un HTML que contiene la figura de Plotly y la tabla resumen."""
    # Generamos sólo el div de la figura (sin <html>…)
    plot_div = to_html(fig, full_html=False, include_plotlyjs='cdn')
    # Montamos el documento completo con la tabla superpuesta
    html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body style="margin:0; padding:0;">
  <div style="
      position:absolute; top:20px; right:20px;
      background:rgba(255,255,255,0.9);
      border:1px solid #ccc; padding:10px;
      max-width:300px; max-height:400px;
      overflow:auto;
      font-family:Arial, sans-serif;
      font-size:12px;
      z-index:100;
    ">
    {summary_html}
  </div>
  {plot_div}
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)



def plot_cm_matrix(event_confusion_matrix, event_idx):
    """Añade la matriz de confusión al figure como una tabla HTML"""
    if event_confusion_matrix is None or event_idx not in event_confusion_matrix:
        return fig

    true_labels = event_confusion_matrix[event_idx]['gen']
    pred_labels = event_confusion_matrix[event_idx]['reco']

    # Obtener etiquetas únicas ordenadas
    labels_true = sorted(set(true_labels))
    labels_pred = sorted(set(pred_labels))

    # Construir la matriz de conteos
    matrix = [[0 for _ in labels_pred] for _ in labels_true]
    for t, p in zip(true_labels, pred_labels):
        if t in labels_true and p in labels_pred:
            i = labels_true.index(t)
            j = labels_pred.index(p)
            matrix[i][j] += 1

    # Construir la tabla HTML
    html = '<table style="border-collapse: collapse; font-size:10px; ' \
           'background-color:rgba(255,255,255,0.9);">'
    # Cabecera: predichos
    html += '<tr><th style="border:1px solid #888; padding:4px;"></th>'
    for p in labels_pred:
        name = pid_map.get(pos_to_PID.get(p, f"PID {p}"), "Non Recognized")
        html += f'<th style="border:1px solid #888; padding:4px;">{name}</th>'
    html += '</tr>'
    # Filas: verdaderos
    for i, t in enumerate(labels_true):
        name_t = real_pid_map.get(t, "False Prediction")
        html += f'<tr><th style="border:1px solid #888; padding:4px;">{name_t}</th>'
        for j, _ in enumerate(labels_pred):
            html += f'<td style="border:1px solid #888; padding:4px; text-align:center;">{matrix[i][j]}</td>'
        html += '</tr>'
    html += '</table>'
    return html

# ...existing code...

def visualize_event(hit_positions, hit_energy, hit_type, hit_id, hits_part_pid,
                    pred_particle_positions, pred_particle_energy, pred_particle_types=None,
                    real_particle_positions=None, real_particle_energy=None, real_particle_types=None,
                    event_confusion_matrix=None, event_idx=0, output_dir="./plots"):
    """Visualiza los hits y las partículas reconstruidas para un evento"""
    
    # Crear figura 3D
    fig = go.Figure()
    
    fig = plot_hits(fig, hit_positions, hit_energy, hit_type, hit_id, hits_part_pid, event_idx)
    fig = plot_reco_particles(
        fig, pred_particle_positions, pred_particle_energy, pred_particle_types, event_idx
    )
    fig = plot_gen_particle(
        fig, real_particle_positions, real_particle_energy, real_particle_types, event_idx
    )
        
    
    cm_summary = plot_cm_matrix(event_confusion_matrix, event_idx)
    
    # Configurar la presentación con Z horizontal
    fig.update_layout(
        title=f"Event {event_idx}",
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='data',
            camera=dict(
                # Esta configuración asegura que Z es el eje horizontal principal
                eye=dict(x=1.5, y=0.0, z=0.0),  # Vista directamente desde el eje X
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)          # Eje Y apunta hacia arriba
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    # Guardar y mostrar
    os.makedirs(output_dir, exist_ok=True)
    save_html_with_summary(
    fig,
    cm_summary, 
    os.path.join(output_dir, f"event_{event_idx}.html")
    )
    return fig

def get_particle_data(batch_data, batch_idx, particle_data: dict, data_type = "reco"):
    """Extrae datos de partículas reconstruidas o generadas"""
    if data_type == "reco":
        # Crear máscaras para filtrar NaNs
        valid_pos_mask = [not np.isnan(p).any() for p in batch_data.pred_pos_matched.values]
        # valid_energy_mask = ~np.isnan(batch_data.calibrated_E.values)
        
        # Aplicar máscaras
        pos = np.array([p for p, valid in zip(batch_data.pred_ref_pt_matched.values, valid_pos_mask) if valid])
        energy = batch_data.calibrated_E.values[valid_pos_mask]
        types = batch_data.pred_pid_matched.values[valid_pos_mask]
        if len(pos) > 0:
            particle_data["pos"][batch_idx] = np.array(pos)
            particle_data["type" ][batch_idx] = types
            particle_data["energy"][batch_idx] = energy
            
    elif data_type == "gen":
        # Crear máscaras para filtrar NaNs
        valid_pos_mask = [not np.isnan(p).any() for p in batch_data.vertex.values]
        # valid_energy_mask = ~np.isnan(batch_data.calibrated_E.values)
        # Aplicar máscaras
        vertex = np.array([p for p, valid in zip(batch_data.vertex.values, valid_pos_mask) if valid])
        distance = np.array([p for p, valid in zip(batch_data.true_pos.values, valid_pos_mask) if valid])
        # La posición es el vertex más la distancia
        # pos = vertex + distance
        pos = distance
        energy = batch_data.true_showers_E.values[valid_pos_mask]
        types = batch_data.pid.values[valid_pos_mask]
        
        if len(pos) > 0:
            particle_data["pos"][batch_idx] = np.array(pos)
            particle_data["type" ][batch_idx] = types
            particle_data["energy"][batch_idx] = energy
    else:
        raise ValueError("Tipo no reconocido. Usa 'reco' o 'gen'.")
    
    return particle_data

def get_event_confusion_matrix(batch_data, batch_idx, event_confusion_matrix: dict):
    event_confusion_matrix.setdefault(batch_idx, {
        'reco': [],
        'gen': []
    })
    for i, row in batch_data.iterrows():
        # Añadir las partículas reconstruidas y generadas

        if np.isnan(row.pred_pos_matched).any():
            pred_pid = -999
        else:
            pred_pid = int(row["pred_pid_matched"])
        # Añadir las partículas generadas
        if np.isnan(row["pid"]) or row["pid"] is None:
            gen_pid = -999
        else:
            gen_pid = int(row["pid"])
        
        event_confusion_matrix[batch_idx]['reco'].append(pred_pid)
        event_confusion_matrix[batch_idx]['gen'].append(gen_pid)
    return event_confusion_matrix


def get_particle_counts(particle_ids, pid_map):
    """Genera un recuento de partículas por tipo"""
    counts = {}
    for pid in particle_ids:
        if np.isnan(pid) if isinstance(pid, float) else False:
            continue
        name = pid_map.get(int(pid), f"Unknown ({int(pid)})")
        counts[name] = counts.get(name, 0) + 1
    
    return counts

def main():
    parser = argparse.ArgumentParser(description='Visualizar hits y clusters')
    parser.add_argument('--tree-file', type=str, required=True, 
                        help='Archivo ROOT con los datos originales')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='Directorio con los archivos de predicciones')
    parser.add_argument('--event', type=int, default=0,
                        help='Índice del evento a visualizar')
    parser.add_argument('--output-dir', type=str, default='./hits_cluster_plots',
                        help='Directorio para guardar las visualizaciones')
    
    args = parser.parse_args()
    
    # Cargar datos del tree original
    hit_x, hit_y, hit_z, hit_energy, hit_type, hit_id, hit_part_pid = load_tree_data(args.tree_file)
    hit_positions = [hit_x, hit_y, hit_z]
    
    # Buscar archivo de predicciones correspondiente al evento
    pred_files = list(Path(args.pred_dir).glob("*_hdbscan_option*_v*.pt"))
    if not pred_files:
        pred_files = list(Path(args.pred_dir).glob("*.pt"))
    if not pred_files:
        print(f"No se encontraron archivos de predicción en {args.pred_dir}")
        return

    print(f"Usando archivo de predicciones: {pred_files[0]}")
    
    # Cargar predicciones
    predictions = load_model_predictions(pred_files[0])

    reco_particle_data = {"pos": {}, "energy": {}, "type": {}}
    gen_particle_data = {"pos": {}, "energy": {}, "type": {}}
    event_confusion_matrix = {}
    # Organizar por número de evento/batch
    for batch_idx in predictions.number_batch.unique():
        batch_data = predictions[predictions.number_batch == batch_idx]
        
        reco_particle_data = get_particle_data(batch_data, batch_idx, reco_particle_data, data_type="reco")
        gen_particle_data = get_particle_data(batch_data, batch_idx, gen_particle_data, data_type="gen")
        event_confusion_matrix = get_event_confusion_matrix(batch_data, batch_idx, event_confusion_matrix)
    
    # Dibujar el evento específico
    visualize_event(
        hit_positions, hit_energy, hit_type,hit_id,hit_part_pid,
        reco_particle_data["pos"], reco_particle_data["energy"], reco_particle_data["type"],
        gen_particle_data["pos"], gen_particle_data["energy"], gen_particle_data["type"],
        event_confusion_matrix=event_confusion_matrix,
        event_idx=args.event,
        output_dir=args.output_dir
    )
    
    print(f"Visualización guardada en {args.output_dir}/event_{args.event}.html")

if __name__ == "__main__":
    main()