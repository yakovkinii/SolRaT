import logging

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from vedo import Arrow, Line, Plotter, Sphere
from yatools import logging_config

from src.multi_term_atom.atomic_data.mock import get_mock_atom_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE

# Initialize logging
logging_config.init(logging.INFO)

# Magnetic field vector B
chi_B = 0
theta_B = pi / 4
BB = 500
Bx, By, Bz = [BB * comp for comp in (np.cos(chi_B) * np.sin(theta_B), np.sin(chi_B) * np.sin(theta_B), np.cos(theta_B))]

# Observation angles
chi, theta, gamma = 0, pi / 4, 0

# Geometry setup
height = 10
vector_length = 10
origin = np.array([0.0, 0.0, 0.0])
B_origin = np.array([0.0, 0.0, height])
B_vec = np.array([Bx, By, Bz])
B_norm = np.linalg.norm(B_vec)
B = B_vec / B_norm

# Projections for angles
B_xy = np.array([Bx, By, 0]) / B_norm
B_z = np.array([0, 0, Bz]) / B_norm
theta_B = np.arctan2(np.linalg.norm(B_xy), np.linalg.norm(B_z))
chi_B = np.arctan2(By, Bx)
omega = np.array([np.cos(chi) * np.sin(theta), np.sin(chi) * np.sin(theta), np.cos(theta)])

logging.info(f"theta_B: {theta_B}, chi_B: {chi_B}")
logging.info(f"theta: {theta}, chi: {chi}, gamma: {gamma}")

# Set up Vedo plotter
plt3d = Plotter(bg="#333377", title="SolRaT")

# Draw volume sphere at B origin
vol_sphere = Sphere(r=2).pos(*B_origin)
plt3d.add(vol_sphere.c("white").alpha(1))

# Draw Sun as large sphere
R_sun = 696340 / 1e3  # in same units
sun = Sphere(r=R_sun, c="#ffbf00").pos(*(origin - np.array([0, 0, R_sun])))
sun.texture(r"C:\pc\SolRaT\earth.jpg")

plt3d.add(sun.alpha(1))

# Draw magnetic field arrow
if B_norm > 1e-3:
    b_arrow = Arrow(start_pt=B_origin, end_pt=B_origin + B * vector_length)
    plt3d.add(b_arrow.c("#7777ff"))

# Draw omega arrow
om_arrow = Arrow(start_pt=B_origin, end_pt=B_origin + omega * vector_length)
plt3d.add(om_arrow.c("white"))

# Draw line from 2*B_origin to origin
line = Line(p0=2 * B_origin, p1=origin)
plt3d.add(line.c("white").lw(1))


# ========== Frequency plots using matplotlib ==========
level_reg, trans_reg, ref_lambdaA, ref_nu = get_mock_atom_data()
nu = np.arange(ref_nu - 1e11, ref_nu + 1e11, 1e8)

# Prepare subplots
fig, axs = plt.subplots(4, 1, sharex=True)
see = MultiTermAtomSEE(level_registry=level_reg, transition_registry=trans_reg)
rte = MultiTermAtomRTE(
    level_registry=level_reg,
    transition_registry=trans_reg,
    nu=nu,
    angles=Angles(
        chi=chi,
        theta=theta,
        gamma=gamma,
        chi_B=chi_B,
        theta_B=theta_B,
    ),
)

rad_tensor = RadiationTensor(transition_registry=trans_reg)
rad_tensor.fill_NLTE_n_w_parametrized(h_arcsec=0.725 * height).rotate_to_magnetic_frame(chi_B=chi_B, theta_B=theta_B)

for Bscale in [0, 0.01, 0.03, 0.1, 0.3, 1]:
    # for Bscale in [1]:
    B_norm_scaled = B_norm * Bscale
    atm_params = AtmosphereParameters(magnetic_field_gauss=B_norm_scaled, delta_v_thermal_cm_sm1=5_000_00)

    see.fill_all_equations(atmosphere_parameters=atm_params, radiation_tensor_in_magnetic_frame=rad_tensor)
    rho = see.get_solution()

    # Compute stokes
    etas = rte.eta_rho_s(
        rho=rho,
        atmosphere_parameters=atm_params,
    )
    norm = np.max(np.abs(etas[0]))

    labels = ["I", "Q", "U", "V"]
    for i, eta in enumerate(etas):
        axs[i].plot(nu, eta / norm, label=f"eta^{{{labels[i]}}} B={B_norm_scaled}")

for ax in axs:
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True)
axs[-1].set_xlabel("Frequency (Hz)")
axs[0].legend()
# plt.tight_layout()


# Focus camera on B_origin and show axes
plt3d.show(axes=1, interactive=False)
plt3d.camera.SetFocalPoint(*B_origin)
plt.show()
