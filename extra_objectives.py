import numpy as np

from time import time
from essos.fields import Vmec as EssosVmec
from essos.dynamics import Tracing, Particles
from essos.constants import FUSION_ALPHA_PARTICLE_ENERGY, ONE_EV

def calculate_loss_fraction(local_wout, stel, s=0.25, tmax=5e-3, B_aries = 5.7, A_aries = 1.7,
                            nparticles=50, trace_tolerance = 1e-4, num_steps_to_plot = 250):
    #### What tmax to set? should it be the same for all particles?
    # This rescaling keeps Larmor radius / Aminor the same between Aries and the current equilibrium
    energy=FUSION_ALPHA_PARTICLE_ENERGY*(stel.wout.volavgB/B_aries)**2*(stel.wout.Aminor_p/A_aries)**2
    # Load equilibrium
    vmec = EssosVmec(local_wout)
    # Set initial particles
    theta = np.linspace(0, 2*np.pi, nparticles)
    phi   = np.linspace(0, 2*np.pi/2/vmec.nfp, nparticles)
    particles = Particles(initial_xyz=np.array([[s]*nparticles, theta, phi]).T, energy=energy, field=vmec)
    # Trace particles
    time0 = time()
    print(f"Tracing {nparticles} particles for {tmax}s with tolerance {trace_tolerance} and energy {energy/ONE_EV} eV for a reactor with B rescaled by a factor {stel.wout.volavgB/B_aries:.2f} and A rescaled by a factor {stel.wout.Aminor_p/A_aries:.2f}")
    tracing = Tracing(field=vmec, model='GuidingCenter', particles=particles, maxtime=tmax, timesteps=num_steps_to_plot, tol_step_size=trace_tolerance)
    # tracing.plot()
    print(f"Final loss fraction: {tracing.loss_fractions[-1]*100:.2f}% took {time()-time0:.2f}s")
