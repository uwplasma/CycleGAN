import os
import argparse
import pandas as pd
import numpy as np
from qsc import Qsc
from qsc.util import fourier_minimum, mu0
from simsopt.mhd import Vmec
from simsopt.mhd import QuasisymmetryRatioResidual
from qi_functions import MaxElongationPen, QuasiIsodynamicResidual, MirrorRatioPen
from desc.equilibrium import Equilibrium
from desc.objectives import get_NAE_constraints
from desc.vmec import VMECIO

def rewrite_vmec_input(filepath: str):
    """Reads a VMEC input file and rewrites specific array lines."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith('NS_ARRAY'):
            new_lines.append('  NS_ARRAY =       51\n')
        elif line.strip().startswith('NITER_ARRAY'):
            new_lines.append('  NITER_ARRAY = 30000\n')
        elif line.strip().startswith('FTOL_ARRAY'):
            new_lines.append('  FTOL_ARRAY =  1E-14\n')
        else:
            new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)

def get_boundary_desc(stel, r=0.1, filename="input.qsc_desc"):
    qsc_eq = stel
    
    print("Generating DESC equilibrium from QSC data...")
    ntheta = 100
    eq_NAE = Equilibrium.from_near_axis(qsc_eq, r=r, L=8, M=8, N=8, ntheta=ntheta)
    constraints = get_NAE_constraints(eq_NAE, qsc_eq, order=2)
    eq_NAE.solve(verbose=3,ftol=1e-2,objective="force",maxiter=25,xtol=1e-6,constraints=constraints)
    # eq_NAE.solve(ftol=1e-4)
    VMECIO.write_vmec_input(eq_NAE, filename)
    rewrite_vmec_input(filename)
    
def get_output_QSC(rc_values, zs_values, nfp_value, etabar_value, B2c_value, p2_value, nphi):
    stel = Qsc(rc=[1.0] + rc_values,zs=[0.0] + zs_values,nfp=nfp_value,
        etabar=etabar_value,B2c=B2c_value,p2=p2_value,order='r2',nphi=nphi)

    axis_length    = stel.axis_length
    iota           = abs(stel.iota)
    max_elongation = stel.max_elongation
    min_L_grad_B   = stel.min_L_grad_B
    min_R0         = stel.min_R0
    r_singularity  = stel.r_singularity
    L_grad_grad_B  = fourier_minimum(stel.L_grad_grad_B)
    B20_variation  = stel.B20_variation
    beta           = -mu0 * p2_value * (stel.r_singularity ** 2) / (stel.B0 ** 2)
    DMerc_times_r2 = stel.DMerc_times_r2

    return stel, [axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity,
            L_grad_grad_B, B20_variation, beta, DMerc_times_r2]
    
def get_output_VMEC(input_vmec_file, verbose=True, helicity_axis=0, mpol=5, ntor=5):
    try:
        stel = Vmec(input_vmec_file, verbose=verbose)
        # stel.indata.mpol = mpol
        # stel.indata.ntor = ntor
        stel.run()
    except Exception as e:
        print(f"Error in VMEC calculation: {e}")
        return [None] * 8

    quasisymmetry_target_surfaces = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
    qs = np.sum(QuasisymmetryRatioResidual(stel, quasisymmetry_target_surfaces,
                                           helicity_m=1, helicity_n=-1*helicity_axis).residuals()**2)
    qi = np.sum(QuasiIsodynamicResidual(stel,[1/16,5/16])**2)
    iota = abs(stel.iota_axis())
    aspect = stel.aspect()
    shear = stel.mean_shear()
    well = stel.vacuum_well()
    elongation = np.max(MaxElongationPen(stel))
    mirror = MirrorRatioPen(stel)
    
    # return [qs, qi, iota, aspect, shear, well, elongation, mirror]
    
    #### GET SURFACE DATA ####
    max_mode = stel.indata.ntor
    surf = stel.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    # surf.fix("rc(0,0)") # Fix major radius to be the same
    dofs = surf.x
    results = np.concatenate(([qs, qi, iota, aspect, shear, well, elongation, mirror], dofs))
    return results

def main(input_QSC_csv, output_csv, output_folder = "output",
         radius=0.05, vmec_input_file="input.qsc", nphi=101,
        #  mpol=5, ntor=7,
         ):
    input_QSC = pd.read_csv(input_QSC_csv)
    
    current_dir = os.getcwd()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_data = []    
    for index, row in input_QSC.iterrows():
        try:
            rc_values = [row['rc1'], row['rc2'], row['rc3']]
            zs_values = [row['zs1'], row['zs2'], row['zs3']]
            nfp_value = int(row['nfp'])
            etabar_value = row['etabar']
            B2c_value = row['B2c']
            p2_value = row['p2']

            stel, qsc_output = get_output_QSC(rc_values, zs_values, nfp_value, etabar_value, B2c_value, p2_value, nphi)
            # radius = stel.r_singularity
            p2_value = row['p2']*radius**2/stel.r_singularity**2
            stel, qsc_output = get_output_QSC(rc_values, zs_values, nfp_value, etabar_value, B2c_value, p2_value, nphi)
            # stel.plot_boundary(r=radius)
            
            os.chdir(output_folder)
            get_boundary_desc(stel, r=radius, filename=vmec_input_file)
            # stel.to_vmec(r=radius, filename=vmec_input_file+'_qsc', ntheta=20,
            #    params=dict(mpol=mpol, ntor=ntor, ns_array=[51], ftol_array=[1e-14], niter_array=[20000]))
            vmec_output = get_output_VMEC(vmec_input_file, abs(stel.helicity))#, mpol=mpol, ntor=ntor)
            os.chdir(current_dir)

            entry = row.tolist() + qsc_output + vmec_output
            output_data.append(entry)
            dof_columns = [f"surf_{i}" for i in range(len(vmec_output) - len(qsc_output))]
            columns = list(input_QSC.columns) + [
                "axis_length", "iota", "max_elongation", "min_L_grad_B", "min_R0", "r_singularity",
                "L_grad_grad_B", "B20_variation", "beta", "DMerc_times_r2",
                "qs", "qi", "vmec_iota", "vmec_aspect", "shear", "well", "elongation", "mirror"
            ] + dof_columns
            # columns = list(input_QSC.columns) + [
            #     "axis_length", "iota", "max_elongation", "min_L_grad_B", "min_R0", "r_singularity",
            #     "L_grad_grad_B", "B20_variation", "beta", "DMerc_times_r2",
            #     "qs", "qi", "vmec_iota", "vmec_aspect", "shear", "well", "elongation", "mirror"
            # ]
            
            
            if all(value is not None for value in vmec_output):
                if not os.path.exists(output_csv):
                    pd.DataFrame([entry], columns=columns).to_csv(output_csv, index=False)
                else:
                    existing_data = pd.read_csv(output_csv)
                    if not existing_data.isin([entry]).all(axis=1).any():
                        pd.DataFrame([entry]).to_csv(output_csv, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Row {index} raised an error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VMEC output from QSC and obtain performance metrics")
    parser.add_argument("--qsc_input_csv", type=str, default="generated_output_fourier.csv", help="Path for input CSV file with pyQSC data")
    parser.add_argument("--qsc_vmec_csv", type=str, default="qsc_vmec_desc.csv", help="Path for output CSV file with VMEC and QSC data")
    args = parser.parse_args()
    main(input_QSC_csv=args.qsc_input_csv, output_csv=args.qsc_vmec_csv)

    # import matplotlib.pyplot as plt
    # output_data = pd.read_csv(args.qsc_vmec_csv)
    # plt.figure(figsize=(8, 6))
    # plt.plot(output_data['iota'], output_data['vmec_iota'], 'o', label='iota vs vmec_iota')
    # plt.xlabel('iota (QSC)')
    # plt.ylabel('iota (VMEC)')
    # plt.show()