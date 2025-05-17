import re
import numpy as np
import subprocess
import os
import shutil
# from time import time
# from essos.fields import Vmec as EssosVmec
# from essos.dynamics import Tracing, Particles
# from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY, ONE_EV

# def calculate_loss_fraction_ESSOS(local_wout, stel, s=0.25, tmax=5e-3, B_aries = 5.7, A_aries = 1.7,
#                             nparticles=50, trace_tolerance = 1e-4, num_steps_to_plot = 250):
#     #### What tmax to set? should it be the same for all particles?
#     # This rescaling keeps Larmor radius / Aminor the same between Aries and the current equilibrium
#     energy=FUSION_ALPHA_PARTICLE_ENERGY*(stel.wout.volavgB/B_aries)**2*(stel.wout.Aminor_p/A_aries)**2
#     # Load equilibrium
#     vmec = EssosVmec(local_wout)
#     # Set initial particles
#     theta = np.linspace(0, 2*np.pi, nparticles)
#     phi   = np.linspace(0, 2*np.pi/2/vmec.nfp, nparticles)
#     particles = Particles(initial_xyz=np.array([[s]*nparticles, theta, phi]).T, energy=energy, field=vmec)
#     # Trace particles
#     time0 = time()
#     print(f"Tracing {nparticles} particles for {tmax}s with tolerance {trace_tolerance} and energy {energy/ONE_EV} eV for a reactor with B rescaled by a factor {stel.wout.volavgB/B_aries:.2f} and A rescaled by a factor {stel.wout.Aminor_p/A_aries:.2f}")
#     tracing = Tracing(field=vmec, model='GuidingCenter', particles=particles, maxtime=tmax, timesteps=num_steps_to_plot, tol_step_size=trace_tolerance)
#     # tracing.plot()
#     print(f"Final loss fraction: {tracing.loss_fractions[-1]*100:.2f}% took {time()-time0:.2f}s")

def calculate_loss_fraction_SIMPLE(local_wout, stel, rank, SIMPLE_executable, SIMPLE_input, SIMPLE_output, s=0.3, trace_time=5e-3, B_aries = 5.7, A_aries = 1.7, nparticles=2000):
    if not os.path.exists(SIMPLE_output):
        SIMPLE_input_to_use = 'simple.in'
        
        current_dir = os.getcwd()
        rank_dir = f"ranks/rank_{rank}"
        os.makedirs(rank_dir, exist_ok=True)
        os.chdir(rank_dir)
        
        B_scale = B_aries/stel.wout.b0
        A_scale = A_aries/stel.wout.Aminor_p
        
        replacements = {
            'ntestpart': f'{nparticles}',
            'trace_time': f'{trace_time}',
            'sbeg': f'{s}',
            'num_surf': 1,
            'netcdffile': f"'{local_wout}'",
            'vmec_B_scale': f'{B_scale}',
            'vmec_RZ_scale': f'{A_scale}',
        }
        
        def update_line(line, key, value):
            """Replace the value of a Fortran namelist key while preserving comments."""
            pattern = rf'^(\s*{key}\s*=\s*)(.*?)(\s*!.*)?$'
            match = re.match(pattern, line)
            if match:
                prefix, _, comment = match.groups()
                comment = comment or ''
                return f"{prefix}{value}{comment}\n"
            return line
        
        with open(SIMPLE_input, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            updated = False
            for key, value in replacements.items():
                if re.match(rf'\s*{key}\s*=', line):
                    new_lines.append(update_line(line, key, value))
                    updated = True
                    break
            if not updated:
                new_lines.append(line)

        with open(SIMPLE_input_to_use, 'w') as f:
            f.writelines(new_lines)
        
        # Run the SIMPLE executable
        result = subprocess.run([SIMPLE_executable, SIMPLE_input_to_use], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running SIMPLE: {result.stderr}")
            return np.nan, np.nan
        shutil.move('confined_fraction.dat', SIMPLE_output)
        os.chdir(current_dir)
        shutil.rmtree(rank_dir)
    
    data = np.loadtxt(SIMPLE_output)
    loss_fraction_times = data[:, 0]
    confined_fraction_passing = data[:, 1]
    confined_fraction_trapped = data[:, 2]
    loss_fraction = 1 - (confined_fraction_passing + confined_fraction_trapped)
    return loss_fraction, loss_fraction_times
    