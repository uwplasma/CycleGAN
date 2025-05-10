import os
import re
import csv
import h5py
import numpy as np
import pandas as pd
from desc.io import load
from desc.vmec import VMECIO
from simsopt.mhd import Vmec, vmec_compute_geometry, QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen

# === Paths ===
GX_zenodo_dir = "/Users/rogeriojorge/Downloads/GX_stellarator_zenodo"
CycleGAN_dir = "/Users/rogeriojorge/local/CycleGAN"
data_folder = "20250119-01-gyrokinetics_machine_learning_zenodo/data_generation_and_analysis"
h5_path = os.path.join(GX_zenodo_dir, "20250102-01_GX_stellarator_dataset.h5")
csv_path = os.path.join(CycleGAN_dir, "stel_results.csv")
parquet_path = os.path.join(CycleGAN_dir, "stel_results.parquet")

# === Load HDF5 Data ===
with h5py.File(h5_path, "r") as f:
    equilibrium_class = f["/equilibrium_class"][()]
    equilibrium_class_descriptions = f["/equilibrium_class_descriptions"][()]
    equilibrium_files = f["/equilibrium_files"][()]
    FSA_grad_xs = f["/FSA_grad_xs"][()]
    fixed_gradient_simulations_Q_avgs = f["/fixed_gradient_simulations/Q_avgs"][()]
    fixed_gradient_simulations_Q_avgs_divided_by_FSA_grad_x = f["/fixed_gradient_simulations/Q_avgs_divided_by_FSA_grad_x"][()]
    fixed_gradient_simulations_Q_avgs_vs_z = f["/fixed_gradient_simulations/Q_avgs_vs_z"][()]
    fixed_gradient_simulations_Q_stds = f["/fixed_gradient_simulations/Q_stds"][()]
    fixed_gradient_simulations_Q_stds_divided_by_FSA_grad_x = f["/fixed_gradient_simulations/Q_stds_divided_by_FSA_grad_x"][()]
    fixed_gradient_simulations_a_over_LT = f["/fixed_gradient_simulations/a_over_LT"][()]
    fixed_gradient_simulations_a_over_Ln = f["/fixed_gradient_simulations/a_over_Ln"][()]
    fixed_gradient_simulations_zonal_phi2_amplitudes = f["/fixed_gradient_simulations/zonal_phi2_amplitudes"][()]
    n_tubes = f["/n_tubes"][()]
    scalar_feature_matrix = f["/scalar_feature_matrix"][()]
    scalar_features = f["/scalar_features"][()]
    scalar_features = [s.decode("utf-8") if isinstance(s, bytes) else s for s in scalar_features]
    varied_gradient_simulations_Q_avgs = f["/varied_gradient_simulations/Q_avgs"][()]
    varied_gradient_simulations_Q_avgs_divided_by_FSA_grad_x = f["/varied_gradient_simulations/Q_avgs_divided_by_FSA_grad_x"][()]
    varied_gradient_simulations_Q_avgs_vs_z = f["/varied_gradient_simulations/Q_avgs_vs_z"][()]
    varied_gradient_simulations_Q_stds = f["/varied_gradient_simulations/Q_stds"][()]
    varied_gradient_simulations_Q_stds_divided_by_FSA_grad_x = f["/varied_gradient_simulations/Q_stds_divided_by_FSA_grad_x"][()]
    varied_gradient_simulations_a_over_LT = f["/varied_gradient_simulations/a_over_LT"][()]
    varied_gradient_simulations_a_over_Ln = f["/varied_gradient_simulations/a_over_Ln"][()]
    varied_gradient_simulations_zonal_phi2_amplitudes = f["/varied_gradient_simulations/zonal_phi2_amplitudes"][()]

n_files = len(equilibrium_files)

old_eq_file_path = ""
for file_index in range(n_files):
    # === Load and Process VMEC Equilibrium ===
    eq_file_relpath = equilibrium_files[file_index].decode("utf-8") if isinstance(equilibrium_files[file_index], bytes) else equilibrium_files[file_index]
    eq_file_path = os.path.join(GX_zenodo_dir, data_folder, eq_file_relpath)
    print(f'Processing index {file_index + 1} of {n_files} with file {eq_file_path}')
    if eq_file_path != old_eq_file_path:
        eq = load(eq_file_path)
        VMECIO.save(eq, 'wout.nc')
        stel = Vmec("wout.nc")
        old_eq_file_path = eq_file_path
        
    # === Compute Diagnostics ===
    target_surfaces = np.linspace(0, 1, 11)
    qa = np.sum(QuasisymmetryRatioResidual(stel, target_surfaces, helicity_m=1, helicity_n=0).residuals()**2)
    qh = np.sum(QuasisymmetryRatioResidual(stel, target_surfaces, helicity_m=1, helicity_n=-1).residuals()**2)
    
    try:
        qi = np.sum(QuasiIsodynamicResidual(stel, [1/16, 5/16])**2)
    except Exception as e:
        print(f"Error calculating qi for file index {file_index}: {e}")
        qi = np.nan
        
    data = vmec_compute_geometry(stel, s=1, theta=np.linspace(0, 2 * np.pi, 50), phi=np.linspace(0, 2 * np.pi, 150))
    L_grad_B_max = np.max(data.L_grad_B)
    L_grad_B_min = np.min(data.L_grad_B)

    # === Extract VMEC Quantities ===
    iota_axis = stel.iota_axis()
    iota_edge = stel.iota_edge()
    mean_iota = stel.mean_iota()
    shear = stel.mean_shear()
    well = stel.vacuum_well()
    elongation = np.max(MaxElongationPen(stel))
    mirror = MirrorRatioPen(stel)
    Dmerc_min = np.min(stel.wout.DMerc[1:])
    Dmerc_max = np.max(stel.wout.DMerc[1:])
    volume = stel.volume()
    betatotal = stel.wout.betatotal
    am_pressure = stel.wout.am
    phiedge = stel.wout.phi[-1]

    # === Extract Boundary Modes ===
    surf = stel.boundary
    formatted_modes = [
        f"rbc_{int(match.group(2))}_{int(match.group(3))}" if match.group(1) == "rc"
        else f"zbs_{int(match.group(2))}_{int(match.group(3))}"
        for name in surf.dof_names if (match := re.search(r":(rc|zs)\(([-\d]+),([-\d]+)\)", name))
    ]

    # === CSV Columns and Values ===
    columns = (
        ['qa', 'qh', 'qi', 'iota_axis', 'iota_edge', 'mean_iota', 'shear', 'well', 'elongation',
        'mirror', 'Dmerc_min', 'Dmerc_max', 'volume', 'betatotal', 'phiedge', 'FSA_grad_xs',
        'fixed_grad_Q', 'fixed_grad_Q_over_FSA_grad_x', 'fixed_grad_Q_std', 'fixed_grad_Q_std_over_FSA_grad_x',
        'fixed_grad_a_over_LT', 'fixed_grad_a_over_Ln', 'fixed_grad_zonal_phi2',
        'varied_grad_Q', 'varied_grad_Q_over_FSA_grad_x', 'varied_grad_Q_std', 'varied_grad_Q_std_over_FSA_grad_x',
        'varied_grad_a_over_LT', 'varied_grad_a_over_Ln', 'varied_grad_zonal_phi2',
        'L_grad_B_max', 'L_grad_B_min'
        ]
        + formatted_modes
        + [f"am_pressure_{i}" for i in range(len(am_pressure))]
        + scalar_features
    )
   
    results = np.concatenate([
        [qa, qh, qi, iota_axis, iota_edge, mean_iota, shear, well, elongation,
        mirror, Dmerc_min, Dmerc_max, volume, betatotal, phiedge, FSA_grad_xs[file_index],
        fixed_gradient_simulations_Q_avgs[file_index],
        fixed_gradient_simulations_Q_avgs_divided_by_FSA_grad_x[file_index],
        fixed_gradient_simulations_Q_stds[file_index],
        fixed_gradient_simulations_Q_stds_divided_by_FSA_grad_x[file_index],
        fixed_gradient_simulations_a_over_LT[file_index],
        fixed_gradient_simulations_a_over_Ln[file_index],
        fixed_gradient_simulations_zonal_phi2_amplitudes[file_index],
        varied_gradient_simulations_Q_avgs[file_index],
        varied_gradient_simulations_Q_avgs_divided_by_FSA_grad_x[file_index],
        varied_gradient_simulations_Q_stds[file_index],
        varied_gradient_simulations_Q_stds_divided_by_FSA_grad_x[file_index],
        varied_gradient_simulations_a_over_LT[file_index],
        varied_gradient_simulations_a_over_Ln[file_index],
        varied_gradient_simulations_zonal_phi2_amplitudes[file_index],
        L_grad_B_max, L_grad_B_min
        ],
        surf.x,
        am_pressure,
        scalar_feature_matrix[file_index]
    ])

    # Convert results to string
    results_str = list(map(str, results))

    # # === Write or Append to CSV ===
    # file_exists = os.path.exists(csv_path)

    # # Open file in write mode if it doesn't exist, or read+write mode if it does
    # with open(csv_path, 'a+', newline='') as f:
    #     f.seek(0)  # Move file pointer to the beginning
    #     reader = csv.reader(f)
    #     rows = list(reader)  # Read all rows to check for duplicates

    #     writer = csv.writer(f)

    #     # If the file is empty or doesn't exist, write the header
    #     if not file_exists or not rows:
    #         writer.writerow(columns)

    #     # Check if results_str already exists, and append if it does not
    #     if results_str not in [row for row in rows]:
    #         writer.writerow(results_str)
            
    # Convert results to DataFrame
    df_new = pd.DataFrame([results], columns=columns)

    # If file exists, load and check for duplicates
    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        
        # Check if new row is a duplicate
        if not (df_existing[columns] == df_new[columns].iloc[0]).all(axis=1).any():
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(parquet_path, index=False)
    else:
        df_new.to_parquet(parquet_path, index=False)