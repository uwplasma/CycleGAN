import os
import re
import gc
import h5py
import psutil
import random
import tracemalloc
import numpy as np
import pandas as pd

from time import time
from mpi4py import MPI
from desc.io import load
from tqdm.auto import tqdm
from desc.vmec import VMECIO
from desc.grid import LinearGrid
from extra_objectives import calculate_loss_fraction_SIMPLE
from simsopt.mhd import RedlGeomVmec
from simsopt.mhd import Vmec, vmec_compute_geometry, QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual
from desc.objectives import (QuasisymmetryTripleProduct, EffectiveRipple, GammaC,
                             Isodynamicity, MagneticWell, MercierStability)
import shutil

# === Paths ===
HOME_DIR = os.path.join(os.environ.get("HOME", os.path.expanduser("~")), "local")
SCRATCH_DIR = os.environ.get("SCRATCH")
GX_zenodo_dir = os.path.join(SCRATCH_DIR,"GX_stellarator_zenodo")
CycleGAN_dir = os.path.join(SCRATCH_DIR,"CycleGAN")
SIMPLE_executable = os.path.join(HOME_DIR,"SIMPLE","build","simple.x")
data_folder = "data_generation_and_analysis"
h5_path = os.path.join(GX_zenodo_dir, "20250102-01_GX_stellarator_dataset.h5")
csv_path = os.path.join(CycleGAN_dir, "stel_results.csv")
wouts_dir = os.path.join(CycleGAN_dir,"wouts")
SIMPLE_input = os.path.join(CycleGAN_dir, "simple_full.in")
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


def compute_DESC_QI_objectives_global_if_not_found(eq_filename, eq, rank, stel, s_targets_qi = [1/16, 5/16, 9/16, 13/16]):
    start_time = time()
    print(f"[Rank {rank}] Computing DESC global objectives for {eq_filename}...")
    grid_global           = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, L=eq.L_grid, axis=False)
    grid_global_sym_False = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False,  L=eq.L_grid, axis=False)
    obj = QuasisymmetryTripleProduct(eq=eq, grid=grid_global);obj.build(verbose=0);qs_tp_global=obj.compute_scalar(*obj.xs(eq))
    obj = Isodynamicity(             eq=eq, grid=grid_global);obj.build(verbose=0);isodynamicity_global=obj.compute_scalar(*obj.xs(eq))
    obj = EffectiveRipple(           eq=eq, grid=grid_global_sym_False, jac_chunk_size=1, num_quad=16, num_well=200, num_transit=20, num_pitch=31);obj.build(verbose=0);effective_ripple_global=obj.compute_scalar(*obj.xs(eq))
    obj = GammaC(                    eq=eq, grid=grid_global_sym_False, jac_chunk_size=1, num_quad=16, num_well=200, num_transit=20, num_pitch=31);obj.build(verbose=0);gamma_c_global=obj.compute_scalar(*obj.xs(eq))
    obj = MercierStability(eq=eq, grid=grid_global);obj.build(verbose=0);mercier_stability_global=obj.compute(*obj.xs(eq))
    DMerc_min = np.min(mercier_stability_global);DMerc_max = np.max(mercier_stability_global)
    print(f"[Rank {rank}] Time taken for DESC global objectives: {time()-start_time:.2f} seconds")
    
    try: qi_global = np.sum(QuasiIsodynamicResidual(stel, s_targets_qi)**2)
    except Exception as e: qi_global = np.nan;print(f"[Rank {rank}] Error calculating qi at eq_filename {eq_filename}: {e}") 
    if qi_global == 0.0: qi_global = np.nan
    
    return (qs_tp_global, effective_ripple_global, gamma_c_global, isodynamicity_global, DMerc_min, DMerc_max, qi_global)
    
def compute_DESC_QI_objectives_global(eq_filename, eq, rank, stel):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        match = df[df["file"] == eq_filename]
        if not match.empty:
            print(f"[Rank {rank}] Found existing DESC and QI global data for {eq_filename}. Using it.")
            qs_tp_global = match.iloc[0]["qs_triple_product_global"]
            effective_ripple_global = match.iloc[0]["effective_ripple_global"]
            gamma_c_global = match.iloc[0]["gamma_c_global"]
            isodynamicity_global = match.iloc[0]["isodynamicity_global"]
            DMerc_min = match.iloc[0]["DMerc_min"]
            DMerc_max = match.iloc[0]["DMerc_max"]
            qi_global = match.iloc[0]["qi_global"]
            return (qs_tp_global, effective_ripple_global, gamma_c_global, isodynamicity_global, DMerc_min, DMerc_max, qi_global)
        else:
            return compute_DESC_QI_objectives_global_if_not_found(eq_filename, eq, rank, stel)
    else:
        return compute_DESC_QI_objectives_global_if_not_found(eq_filename, eq, rank, stel)

def compute_DESC_QI_objectives(eq_filename, eq, rank, stel, rho):
    (qs_tp_global, effective_ripple_global, gamma_c_global, isodynamicity_global,
     DMerc_min, DMerc_max, qi_global) = compute_DESC_QI_objectives_global(eq_filename, eq, rank, stel)

    start_time = time()
    print(f"[Rank {rank}] Computing DESC local objectives for {eq_filename}...")
    grid_this_rho           = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    grid_this_rho_sym_False = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False,  rho=rho)
    obj = QuasisymmetryTripleProduct(eq=eq, grid=grid_this_rho);obj.build(verbose=0);qs_tp_this_rho=obj.compute_scalar(*obj.xs(eq))
    obj = Isodynamicity(             eq=eq, grid=grid_this_rho);obj.build(verbose=0);isodynamicity_this_rho=obj.compute_scalar(*obj.xs(eq))
    obj = EffectiveRipple(           eq=eq, grid=grid_this_rho_sym_False, jac_chunk_size=1, num_quad=16, num_well=200, num_transit=20, num_pitch=31);obj.build(verbose=0);effective_ripple_this_rho=obj.compute_scalar(*obj.xs(eq))
    obj = GammaC(                    eq=eq, grid=grid_this_rho_sym_False, jac_chunk_size=1, num_quad=16, num_well=200, num_transit=20, num_pitch=31);obj.build(verbose=0);gamma_c_this_rho=obj.compute_scalar(*obj.xs(eq))
    obj = MagneticWell(eq=eq, grid=grid_this_rho);obj.build(verbose=0);magnetic_well_this_rho=obj.compute(*obj.xs(eq))[0]
    obj = MercierStability(eq=eq, grid=grid_this_rho);obj.build(verbose=0);DMerc_this_rho=obj.compute(*obj.xs(eq))[0]
    print(f"[Rank {rank}] Time taken for DESC local objectives: {time()-start_time:.2f} seconds")

    try: qi_this_rho = np.sum(QuasiIsodynamicResidual(stel, [rho**2])**2)
    except Exception as e: qi_this_rho = np.nan;print(f"[Rank {rank}] Error calculating qi at eq_filename {eq_filename}: {e}") 
    
    return (qs_tp_global, qs_tp_this_rho, effective_ripple_global, effective_ripple_this_rho,
            gamma_c_global, gamma_c_this_rho, isodynamicity_global, isodynamicity_this_rho,
            magnetic_well_this_rho, DMerc_this_rho, DMerc_min, DMerc_max, qi_global, qi_this_rho)
    
def process_equilibrium(i, eq_relpath, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data, rho, eq_filename):
    eq_path = os.path.join(GX_zenodo_dir, data_folder, eq_relpath)
    local_wout = os.path.join(wouts_dir, f"wout_{eq_filename}.nc")
    
    eq = load(eq_path)
    # eq.change_resolution(M=4, N=4)
    # eq.surface = eq.get_surface_at(rho=1.0)
    
    # Only save the equilibrium if it hasn't been processed yet
    if not os.path.exists(local_wout):
        start_time = time()
        print(f"[Rank {rank}] Saving VMEC output for {eq_filename}...")
        VMECIO.save(eq, local_wout, verbose=0)
        print(f"[Rank {rank}] Saved equilibrium to {local_wout} in {time()-start_time:.2f} seconds")
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
    stel = Vmec(local_wout, verbose=False)
    
    (qs_tp_global, qs_tp_this_rho, effective_ripple_global, effective_ripple_this_rho,
     gamma_c_global, gamma_c_this_rho, isodynamicity_global, isodynamicity_this_rho,
     magnetic_well_this_rho, DMerc_this_rho, DMerc_min, DMerc_max,
     qi_global, qi_this_rho) = compute_DESC_QI_objectives(eq_filename, eq, rank, stel, rho)

    s_targets_qs = np.linspace(0, 1, 5)
    qa_global = np.sum(QuasisymmetryRatioResidual(stel, s_targets_qs, helicity_m=1, helicity_n=0).residuals()**2)
    qh_global = np.sum(QuasisymmetryRatioResidual(stel, s_targets_qs, helicity_m=1, helicity_n=-1).residuals()**2)
    qp_global = np.sum(QuasisymmetryRatioResidual(stel, s_targets_qs, helicity_m=0, helicity_n=1).residuals()**2)

    qa_this_rho = np.sum(QuasisymmetryRatioResidual(stel, [rho**2], helicity_m=1, helicity_n=0).residuals()**2)
    qh_this_rho = np.sum(QuasisymmetryRatioResidual(stel, [rho**2], helicity_m=1, helicity_n=-1).residuals()**2)
    qp_this_rho = np.sum(QuasisymmetryRatioResidual(stel, [rho**2], helicity_m=0, helicity_n=1).residuals()**2)

    geom = vmec_compute_geometry(stel, s=1, theta=np.linspace(0, 2*np.pi, 50), phi=np.linspace(0, 2*np.pi, 150))
    L_grad_B_max_surface = np.max(geom.L_grad_B)
    L_grad_B_min_surface = np.min(geom.L_grad_B)

    geom_this_rho = vmec_compute_geometry(stel, s=rho**2, theta=np.linspace(0, 2*np.pi, 50), phi=np.linspace(0, 2*np.pi, 150))
    L_grad_B_max_this_rho = np.max(geom_this_rho.L_grad_B)
    L_grad_B_min_this_rho = np.min(geom_this_rho.L_grad_B)
    
    start_time = time()
    SIMPLE_output = os.path.join(wouts_dir, f"simple_output_{eq_filename}.dat")
    print(f"[Rank {rank}] Running SIMPLE for {eq_filename}...")
    loss_fraction, loss_fraction_times = calculate_loss_fraction_SIMPLE(local_wout=local_wout, stel=stel, SIMPLE_output=SIMPLE_output,
                                                    SIMPLE_executable=SIMPLE_executable, SIMPLE_input=SIMPLE_input, rank=rank)
    loss_fraction_3em5 = loss_fraction[np.argmin(np.abs(loss_fraction_times - 3e-5))]
    loss_fraction_1em4 = loss_fraction[np.argmin(np.abs(loss_fraction_times - 1e-4))]
    loss_fraction_1em3 = loss_fraction[np.argmin(np.abs(loss_fraction_times - 1e-3))]
    loss_fraction_5em3 = loss_fraction[np.argmin(np.abs(loss_fraction_times - 5e-3))]
    loss_fraction_1em2 = loss_fraction[np.argmin(np.abs(loss_fraction_times - 1e-2))]
    print(f"[Rank {rank}] Loss fraction at 3e-5 = {loss_fraction_3em5}, at 1e-4 = {loss_fraction_1em4}, at 1e-3 = {loss_fraction_1em3}, at 5e-3 = {loss_fraction_5em3} and 1e-2  = {loss_fraction_1em2}. Calculation took {time()-start_time:.2f} seconds")

    stru = RedlGeomVmec(vmec=stel, surfaces=[0.001,rho**2,1])()
    mirror_ratio_axis     = (stru.Bmax[0]-stru.Bmin[0])/(stru.Bmax[0]+stru.Bmin[0])
    mirror_ratio_this_rho = (stru.Bmax[1]-stru.Bmin[1])/(stru.Bmax[1]+stru.Bmin[1])
    mirror_ratio_surface  = (stru.Bmax[2]-stru.Bmin[2])/(stru.Bmax[2]+stru.Bmin[2])
    trapped_fraction_axis     = stru.f_t[0]
    trapped_fraction_this_rho = stru.f_t[1]
    trapped_fraction_surface  = stru.f_t[2]
    Boozer_G = stru.G[0]

    results = [qa_global, qh_global, qp_global, qi_global, Boozer_G, trapped_fraction_axis, trapped_fraction_surface,
               qs_tp_global, effective_ripple_global, gamma_c_global, isodynamicity_global,
               loss_fraction_3em5, loss_fraction_1em4, loss_fraction_1em3, loss_fraction_5em3, loss_fraction_1em2,
               stel.iota_axis(), stel.iota_edge(), stel.mean_iota(), stel.mean_shear(),
               stel.vacuum_well(), np.max(MaxElongationPen(stel)),
               mirror_ratio_axis, mirror_ratio_surface,
               DMerc_min, DMerc_max, stel.wout.Aminor_p, stel.wout.Rmajor_p,
               stel.volume(), stel.wout.betatotal, stel.wout.betaxis, stel.wout.volavgB, stel.wout.phi[-1], FSA_grad_xs[i],
               qa_this_rho, qh_this_rho, qp_this_rho, qi_this_rho,
               qs_tp_this_rho, effective_ripple_this_rho, gamma_c_this_rho, isodynamicity_this_rho,
               mirror_ratio_this_rho, trapped_fraction_this_rho, magnetic_well_this_rho, DMerc_this_rho,
               L_grad_B_max_surface, L_grad_B_min_surface, L_grad_B_max_this_rho, L_grad_B_min_this_rho]

    for key in fixed_data: results.append(fixed_data[key][i])
    for key in varied_data: results.append(varied_data[key][i])

    surf = stel.boundary
    surf.change_resolution(mpol=4, ntor=4) # Force every surface to have the same resolution
    mode_names = [f"rbc_{int(m.group(2))}_{int(m.group(3))}" if m.group(1) == "rc" else f"zbs_{int(m.group(2))}_{int(m.group(3))}"
                  for name in surf.dof_names if (m := re.search(r":(rc|zs)\(([-\d]+),([-\d]+)\)", name))]
    results += list(surf.x)
    results += list(stel.wout.am[:4])
    results += list(scalar_feature_matrix[i])

    all_columns = (
        ['qa_global', 'qh_global', 'qp_global', 'qi_global', 'Boozer_G', 'trapped_fraction_axis', 'trapped_fraction_surface',
         'qs_triple_product_global', 'effective_ripple_global', 'gamma_c_global', 'isodynamicity_global',
         'loss_fraction_3e-5s', 'loss_fraction_1e-4s', 'loss_fraction_1e-3s', 'loss_fraction_5e-3s', 'loss_fraction_1e-2s',
         'iota_axis', 'iota_edge', 'mean_iota', 'mean_shear', 'magnetic_well_global', 'max_elongation',
         'mirror_ratio_axis', 'mirror_ratio_surface', 'DMerc_min', 'DMerc_max', 'Aminor', 'Rmajor', 'volume', 'betatotal', 'betaxis',
         'volavgB', 'phiedge', 'FSA_grad_xs',
         'qa_this_rho', 'qh_this_rho', 'qp_this_rho', 'qi_this_rho',
         'qs_triple_product_this_rho', 'effective_ripple_this_rho', 'gamma_c_this_rho', 'isodynamicity_this_rho',
         'mirror_ratio_this_rho', 'trapped_fraction_this_rho', 'magnetic_well_this_rho', 'DMerc_this_rho',
         'L_grad_B_max_surface', 'L_grad_B_min_surface','L_grad_B_max_this_rho', 'L_grad_B_min_this_rho'] +
        [f"fixed_grad_{k}" for k in fixed_data] +
        [f"varied_grad_{k}" for k in varied_data] +
        mode_names +
        [f"am_pressure_{j}" for j in range(len(stel.wout.am[:4]))] +
        scalar_features)
    df_row = pd.DataFrame([results], columns=all_columns)
    
    # Reorder columns: move 'file' to the front
    df_row["file"] = eq_filename
    cols = ["file"] + [col for col in df_row.columns if col != "file"]
    df_row = df_row[cols]
    
    del stel, results, eq, surf
    
    return df_row

def is_duplicate_row(existing_df: pd.DataFrame, new_row: pd.Series, tol: float = 1e-6) -> bool:
    # Align row to match existing_df columns
    new_row = new_row[existing_df.columns]

    # Identify numeric and non-numeric columns
    numeric_cols = existing_df.select_dtypes(include=[np.number]).columns
    other_cols = [col for col in existing_df.columns if col not in numeric_cols]

    # Convert both to float to ensure compatibility
    try:
        existing_numeric = existing_df[numeric_cols].astype(float).values
        row_numeric = new_row[numeric_cols].astype(float).values
    except Exception as e:
        raise ValueError(f"Numeric conversion failed: {e}")

    # Compare numeric values within tolerance
    numeric_match = np.all(np.isclose(existing_numeric, row_numeric, atol=tol, rtol=0), axis=1)

    # Compare non-numeric columns exactly
    if other_cols:
        existing_others = existing_df[other_cols].astype(str).values
        row_others = new_row[other_cols].astype(str).values
        other_match = np.all(existing_others == row_others, axis=1)
    else:
        other_match = np.full(len(existing_df), True)

    return np.any(numeric_match & other_match)

def main():
    tracemalloc.start()
    if rank==0: memory_report("Start")

    eq_classes, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data = load_static_data()
    rho_index = scalar_features.index("rho")

    # === Round-robin distribution of all eq_classes across ranks ===
    num_files = len(eq_classes)
    files_per_rank = num_files // size  # how many files each rank should process
    # === Indices this rank will process ===
    my_indices = [i for i in range(rank * files_per_rank, (rank + 1) * files_per_rank)]
    # If the division isn't perfect, last rank gets the remaining files
    if rank == size - 1: my_indices.extend(range(size * files_per_rank, num_files))
    # random.shuffle(my_indices)

    # === Main loop for processing ===
    stel_ind = 0
    # for i in my_indices:
    if rank == 0: iterator = tqdm(my_indices, desc=f"Rank {rank}")#, position=rank, leave=True)
    else: iterator = my_indices  # no tqdm for other ranks
    for i in iterator:
        eq_relpath = eq_classes[i].decode() if isinstance(eq_classes[i], bytes) else eq_classes[i]
        try:
            rho = scalar_feature_matrix[i][rho_index] # sqrt(s)
            eq_filename = os.path.basename(eq_relpath).replace(".h5", "")
            # Check if the equilibrium has already been processed
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if "file" in df.columns and "rho" in df.columns:
                    if not df[(df["file"] == eq_filename) & (np.isclose(df["rho"], rho, atol=1e-6))].empty:
                        print(f"[Rank {rank}] Skipping {eq_filename} at rho={rho}: already exists in CSV")
                        continue
            memory_report(f"#{stel_ind+1}/{len(my_indices)}:{i+1}: {eq_relpath[12:]} and rho = {rho}")
            df_row = process_equilibrium(i, eq_relpath, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data, rho, eq_filename)
            if not os.path.exists(csv_path):
                df_row.to_csv(csv_path, mode='w', header=True, index=False)
            else:
                try:
                    if not is_duplicate_row(pd.read_csv(csv_path), df_row.iloc[0], tol=1e-6):
                        print(f"[Rank {rank}] Writing new row to CSV")
                        df_row.to_csv(csv_path, mode='a', header=False, index=False)
                    else:
                        print(f"[Rank {rank}] Skipping duplicate row")
                except Exception as e:
                    print(f"[Rank {rank}] Error checking for duplicate row: {e}")
        except Exception as e: print(f"[Rank {rank}] ERROR at index {i}: {e}")
        gc.collect()
        stel_ind += 1
    current, peak = tracemalloc.get_traced_memory()
    if rank==0: print(f"[Rank {rank}] Final memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB")
    tracemalloc.stop()

if __name__ == "__main__":
    main()
