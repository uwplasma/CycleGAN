#### ADD LOSS FRACTIONS FROM S=0.25 WITH ESSOS and LE/LI from MONKES

import os
import re
import gc
import h5py
import time
import psutil
import datetime
import tracemalloc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

from glob import glob
from mpi4py import MPI
from desc.io import load
from desc.vmec import VMECIO
from simsopt.mhd import Vmec, vmec_compute_geometry, QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen

# === Paths ===
GX_zenodo_dir = "/Users/rogeriojorge/Downloads/GX_stellarator_zenodo"
CycleGAN_dir = "/Users/rogeriojorge/local/CycleGAN"
data_folder = "20250119-01-gyrokinetics_machine_learning_zenodo/data_generation_and_analysis"
h5_path = os.path.join(GX_zenodo_dir, "20250102-01_GX_stellarator_dataset.h5")
parquet_path = os.path.join(CycleGAN_dir, "stel_results.parquet")
merge_interval = 60

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [Rank {rank}] {msg}", flush=True)

def memory_report(prefix=""):
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024 / 1024
    print(f"[Rank {rank}] {prefix} Memory usage: {mem_MB:.2f} MB", flush=True)

def load_static_data():
    with h5py.File(h5_path, "r") as f:
        eq_classes = f["/equilibrium_files"][()]
        scalar_features = [s.decode("utf-8") for s in f["/scalar_features"][()]]
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

def process_equilibrium(i, eq_relpath, old_eq_path, eq_classes, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data):
    eq_path = os.path.join(GX_zenodo_dir, data_folder, eq_relpath)
    print(f"[Rank {rank}] Processing {i+1}/{len(eq_classes)}: {eq_relpath}", flush=True)

    local_wout = f"wout_rank{rank}.nc"
    # Only load and save the equilibrium if it hasn't been processed yet by this rank
    if eq_path != old_eq_path:
        eq = load(eq_path)
        VMECIO.save(eq, local_wout, verbose=0)
        old_eq_path = eq_path

    stel = Vmec(local_wout)

    s_targets = np.linspace(0, 1, 11)
    qa = np.sum(QuasisymmetryRatioResidual(stel, s_targets, helicity_m=1, helicity_n=0).residuals()**2)
    qh = np.sum(QuasisymmetryRatioResidual(stel, s_targets, helicity_m=1, helicity_n=-1).residuals()**2)
    qp = np.sum(QuasisymmetryRatioResidual(stel, s_targets, helicity_m=0, helicity_n=-1).residuals()**2)
    try:
        qi = np.sum(QuasiIsodynamicResidual(stel, [1/16, 5/16])**2)
    except Exception as e:
        print(f"[Rank {rank}] Error calculating qi at index {i}: {e}")
        qi = np.nan

    geom = vmec_compute_geometry(stel, s=1, theta=np.linspace(0, 2*np.pi, 50), phi=np.linspace(0, 2*np.pi, 150))
    L_grad_B_max = np.max(geom.L_grad_B)
    L_grad_B_min = np.min(geom.L_grad_B)

    results = [
        qa, qh, qp, qi,
        stel.iota_axis(), stel.iota_edge(), stel.mean_iota(), stel.mean_shear(),
        stel.vacuum_well(), np.max(MaxElongationPen(stel)), MirrorRatioPen(stel),
        np.min(stel.wout.DMerc[1:]), np.max(stel.wout.DMerc[1:]),
        stel.volume(), stel.wout.betatotal, stel.wout.phi[-1],
        FSA_grad_xs[i]
    ]

    for key in fixed_data:
        results.append(fixed_data[key][i])
    for key in varied_data:
        results.append(varied_data[key][i])

    results += [L_grad_B_max, L_grad_B_min]

    surf = stel.boundary
    mode_names = [
        f"rbc_{int(m.group(2))}_{int(m.group(3))}" if m.group(1) == "rc" else f"zbs_{int(m.group(2))}_{int(m.group(3))}"
        for name in surf.dof_names if (m := re.search(r":(rc|zs)\(([-\d]+),([-\d]+)\)", name))
    ]
    results += list(surf.x)

    results += list(stel.wout.am)
    results += list(scalar_feature_matrix[i])

    all_columns = (
        ['qa', 'qh', 'qp', 'qi', 'iota_axis', 'iota_edge', 'mean_iota', 'shear', 'well', 'elongation',
         'mirror', 'Dmerc_min', 'Dmerc_max', 'volume', 'betatotal', 'phiedge', 'FSA_grad_xs'] +
        [f"fixed_grad_{k}" for k in fixed_data] +
        [f"varied_grad_{k}" for k in varied_data] +
        ['L_grad_B_max', 'L_grad_B_min'] +
        mode_names +
        [f"am_pressure_{j}" for j in range(len(stel.wout.am))] +
        scalar_features
    )
    df_row = pd.DataFrame([results], columns=all_columns)

    # # Clean up wout file
    # os.remove(local_wout)
    return df_row, old_eq_path

def maybe_merge_parquet_files(merged_file_path):
    parquet_files = glob("*rank*.parquet")
    if not parquet_files: return
    print(f"[Rank 0] Merging {len(parquet_files)} parquet files...")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_parquet(merged_file_path, index=False)
    print(f"[Rank 0] Wrote merged file to {merged_file_path}")

def main():
    last_merge_time = time.time()
    tracemalloc.start()
    memory_report("Start")

    eq_classes, scalar_features, scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data = load_static_data()
    old_eq_path = ""

    # === Group indices by unique equilibrium file ===
    eq_to_indices = defaultdict(list)
    for idx, eq_relpath in enumerate(eq_classes):
        key = eq_relpath.decode() if isinstance(eq_relpath, bytes) else eq_relpath
        eq_to_indices[key].append(idx)

    # === Distribute groups of indices (same eq_class) across ranks ===
    eq_keys_sorted = sorted(eq_to_indices.keys())
    eq_chunks = [eq_keys_sorted[i::size] for i in range(size)]  # round-robin allocation

    # === Indices this rank will process ===
    my_indices = []
    for eq_key in eq_chunks[rank]:
        my_indices.extend(eq_to_indices[eq_key])

    # === Main loop for processing ===
    for i in my_indices:
        eq_relpath = eq_classes[i].decode() if isinstance(eq_classes[i], bytes) else eq_classes[i]
        try:
            df_row, old_eq_path = process_equilibrium(
                i, eq_relpath, old_eq_path, eq_classes, scalar_features,
                scalar_feature_matrix, FSA_grad_xs, fixed_data, varied_data
            )
            # Save a separate file per rank to avoid write collisions
            local_parquet = parquet_path.replace(".parquet", f"_rank{rank}.parquet")
            table = pa.Table.from_pandas(df_row)
            if not os.path.exists(local_parquet):
                pq.write_table(table, local_parquet, compression="zstd")
            else:
                with pq.ParquetWriter(local_parquet, table.schema, compression="zstd", use_dictionary=True) as writer:
                    writer.write_table(table)
        except Exception as e:
            print(f"[Rank {rank}] ERROR at index {i}: {e}", flush=True)
        memory_report(f"After index {i}")
        gc.collect()
        
        # Periodically merge
        if rank == 0 and (time.time() - last_merge_time) > merge_interval:
            maybe_merge_parquet_files(merged_file_path=parquet_path)
            last_merge_time = time.time()

    # Remove all wout files created by this rank
    local_wout_files = glob(f"wout_rank{rank}*.nc")
    for wout_file in local_wout_files:
        try:
            os.remove(wout_file)
            print(f"[Rank {rank}] Removed {wout_file}", flush=True)
        except Exception as e:
            print(f"[Rank {rank}] Error removing {wout_file}: {e}", flush=True)

    current, peak = tracemalloc.get_traced_memory()
    print(f"[Rank {rank}] Final memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB", flush=True)
    tracemalloc.stop()

if __name__ == "__main__":
    main()
