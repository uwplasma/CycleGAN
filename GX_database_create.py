import os
import re
import gc
import h5py
import psutil
import tracemalloc
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from mpi4py import MPI
from desc.io import load
from desc.vmec import VMECIO
from extra_objectives import calculate_loss_fraction
from simsopt.mhd import Vmec, vmec_compute_geometry, QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen
import shutil

# === Paths ===
GX_zenodo_dir = "/Users/rogeriojorge/Downloads/GX_stellarator_zenodo"
CycleGAN_dir = "/Users/rogeriojorge/local/CycleGAN"
data_folder = "20250119-01-gyrokinetics_machine_learning_zenodo/data_generation_and_analysis"
h5_path = os.path.join(GX_zenodo_dir, "20250102-01_GX_stellarator_dataset.h5")
csv_path = os.path.join(CycleGAN_dir, "stel_results.csv")
wouts_dir = os.path.join(CycleGAN_dir,"wouts")
os.makedirs(wouts_dir, exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def memory_report(prefix=""):
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024 / 1024
    print(f"[Rank {rank}] {prefix}. Memory: {mem_MB:.2f} MB")
    # if rank==0: tqdm.write(f"[Rank {rank}] {prefix}. Memory: {mem_MB:.2f} MB", end="")

def load_static_data():
    with h5py.File(h5_path, "r") as f:
        eq_classes = f["/equilibrium_files"][()]
        scalar_features = [s.decode("utf-8") for s in f["/scalar_features"][()]]
        rename_map = {"iota": "iota_this_rho", "shat": "shat_this_rho", "aspect/rho": "aspect_ratio_over_rho",
                      "d_pressure_d_s": "d_pressure_d_s_this_rho", "aspect": "aspect_ratio"}
        scalar_features = [rename_map.get(s, s) for s in scalar_features]
        scalar_feature_matrix = f["/scalar_feature_matrix"][()]
        FSA_grad_xs = f["/FSA_grad_xs"][()]
        fixed_keys = [
            "Q_avgs", "Q_avgs_divided_by_FSA_grad_x", "Q_stds",
            "Q_stds_divided_by_FSA_grad_x", "a_over_LT", "a_over_Ln",
            "zonal_phi2_amplitudes"
        ]
        fixed_data = {key: f[f"/fixed_gradient_simulations/{key}"][()] for key in fixed_keys}
        varied_data = {key: f[f"/varied_gradient_simulations/{key}"][()] for key in fixed_keys}
    return eq_classes, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data

def process_equilibrium(i, eq_relpath, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data):
    eq_path = os.path.join(GX_zenodo_dir, data_folder, eq_relpath)
    eq_filename = os.path.basename(eq_relpath).replace(".h5", "")
    local_wout = os.path.join(wouts_dir, f"wout_{eq_filename}.nc")
    
    # Only load and save the equilibrium if it hasn't been processed yet
    if not os.path.exists(local_wout):
        eq = load(eq_path)
        # eq.change_resolution(M=4, N=4)
        # eq.surface = eq.get_surface_at(rho=1.0)
        VMECIO.save(eq, local_wout, verbose=0)
        # current_dir = os.getcwd()
        # os.chdir(wouts_dir)
        # local_vmec_input = local_wout.replace(".nc", "").replace("wout_", "input.")
        # VMECIO.write_vmec_input(eq, local_vmec_input, verbose=0, NS_ARRAY=[101], NITER_ARRAY=[20000], FTOL_ARRAY=[1e-14])
        # print(f"[Rank {rank}] Running VMEC input {local_vmec_input}")
        # stel = Vmec(local_vmec_input, verbose=True)
        # stel.run()
        # shutil.move(stel.output_file, local_wout)
        # shutil.move(stel.input_file+'_000_000000',local_vmec_input)
        # os.remove(local_wout.replace(".nc", "").replace("wout_", "threed1."))
        # os.chdir(current_dir)
        del eq
    stel = Vmec(local_wout, verbose=False)

    s_targets_qs = np.linspace(0, 1, 5)
    qa = np.sum(QuasisymmetryRatioResidual(stel, s_targets_qs, helicity_m=1, helicity_n=0).residuals()**2)
    qh = np.sum(QuasisymmetryRatioResidual(stel, s_targets_qs, helicity_m=1, helicity_n=-1).residuals()**2)
    qp = np.sum(QuasisymmetryRatioResidual(stel, s_targets_qs, helicity_m=0, helicity_n=1).residuals()**2)
    
    s_targets_qi = [1/16, 5/16, 9/16]
    try: qi = np.sum(QuasiIsodynamicResidual(stel, s_targets_qi)**2)
    except Exception as e: qi = np.nan;print(f"[Rank {rank}] Error calculating qi at index {i}: {e}"); 

    geom = vmec_compute_geometry(stel, s=1, theta=np.linspace(0, 2*np.pi, 50), phi=np.linspace(0, 2*np.pi, 150))
    L_grad_B_max = np.max(geom.L_grad_B)
    L_grad_B_min = np.min(geom.L_grad_B)
    
    # loss_fraction = calculate_loss_fraction(local_wout, stel)

    results = [qa, qh, qp, qi,# loss_fraction,
        stel.iota_axis(), stel.iota_edge(), stel.mean_iota(), stel.mean_shear(),
        stel.vacuum_well(), np.max(MaxElongationPen(stel)), MirrorRatioPen(stel),
        np.min(stel.wout.DMerc[4:]), np.max(stel.wout.DMerc[4:]),
        stel.volume(), stel.wout.betatotal, stel.wout.phi[-1], FSA_grad_xs[i]]

    for key in fixed_data: results.append(fixed_data[key][i])
    for key in varied_data: results.append(varied_data[key][i])

    results += [L_grad_B_max, L_grad_B_min]

    surf = stel.boundary
    surf.change_resolution(mpol=8, ntor=8) # Force every surface to have the same resolution
    mode_names = [f"rbc_{int(m.group(2))}_{int(m.group(3))}" if m.group(1) == "rc" else f"zbs_{int(m.group(2))}_{int(m.group(3))}"
                  for name in surf.dof_names if (m := re.search(r":(rc|zs)\(([-\d]+),([-\d]+)\)", name))]
    results += list(surf.x)
    results += list(stel.wout.am[:4])
    results += list(scalar_feature_matrix[i])

    all_columns = (
        ['qa', 'qh', 'qp', 'qi',# 'loss_fraction',
         'iota_axis', 'iota_edge', 'mean_iota', 'mean_shear', 'well', 'elongation',
         'mirror', 'Dmerc_min', 'Dmerc_max', 'volume', 'betatotal', 'phiedge', 'FSA_grad_xs'] +
        [f"fixed_grad_{k}" for k in fixed_data] +
        [f"varied_grad_{k}" for k in varied_data] +
        ['L_grad_B_max', 'L_grad_B_min'] +
        mode_names +
        [f"am_pressure_{j}" for j in range(len(stel.wout.am[:4]))] +
        scalar_features)
    df_row = pd.DataFrame([results], columns=all_columns)
    
    # Reorder columns: move 'file' to the front
    df_row["file"] = eq_filename
    cols = ["file"] + [col for col in df_row.columns if col != "file"]
    df_row = df_row[cols]
    
    del stel, results
    
    return df_row

def main():
    tracemalloc.start()
    if rank==0: memory_report("Start")

    eq_classes, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data = load_static_data()

    # === Round-robin distribution of all eq_classes across ranks ===
    num_files = len(eq_classes)
    files_per_rank = num_files // size  # how many files each rank should process
    # === Indices this rank will process ===
    my_indices = [i for i in range(rank * files_per_rank, (rank + 1) * files_per_rank)]
    # If the division isn't perfect, last rank gets the remaining files
    if rank == size - 1: my_indices.extend(range(size * files_per_rank, num_files))

    # === Main loop for processing ===
    stel_ind = 0
    # for i in my_indices:
    if rank == 0: iterator = tqdm(my_indices, desc=f"Rank {rank}")#, position=rank, leave=True)
    else: iterator = my_indices  # no tqdm for other ranks
    for i in iterator:
        eq_relpath = eq_classes[i].decode() if isinstance(eq_classes[i], bytes) else eq_classes[i]
        try:
            memory_report(f"#{stel_ind+1}/{len(my_indices)}:{i+1}: {eq_relpath[12:]}")
            df_row = process_equilibrium(i, eq_relpath, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data)
            if not os.path.exists(csv_path): df_row.to_csv(csv_path, mode='w', header=True, index=False)
            else: df_row.to_csv(csv_path, mode='a', header=False, index=False)
        except Exception as e: print(f"[Rank {rank}] ERROR at index {i}: {e}")
        gc.collect()
        stel_ind += 1
    current, peak = tracemalloc.get_traced_memory()
    if rank==0: print(f"[Rank {rank}] Final memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB")
    tracemalloc.stop()

if __name__ == "__main__":
    main()
