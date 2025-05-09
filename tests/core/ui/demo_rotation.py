import logging

import numpy as np
import pyvista as pv
import vtk
from matplotlib import pyplot as plt
from numpy import pi
from yatools import logging_config

from src.core.physics.rotations import WignerD, rotate_J
from src.two_term_atom.atomic_data.mock import get_mock_atom_data
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom

_vtk = vtk  # This is a hack for Windows to enable LaTeX rendering in PyVista

logging_config.init(logging.INFO)

# Vector B
Bx, By, Bz = 0, 10000, 10000
chi = 0
theta = pi / 2
gamma = 0


height = 10
vector_length = 10

# Origin and projection
origin = np.array([0.0, 0, 0])
B_origin = np.array([0.0, 0, height])
B_norm = np.linalg.norm([Bx, By, Bz])
B = np.array([Bx, By, Bz]) / B_norm
B_proj_xy = np.array([Bx, By, 0]) / B_norm
B_proj_z = np.array([0, 0, Bz]) / B_norm

# Compute angles
theta_B = np.arctan2(np.linalg.norm(B_proj_xy), np.linalg.norm(B_proj_z))
chi_B = np.arctan2(By, Bx)


logging.info(f"theta_B: {theta_B}, chi_B: {chi_B}")
logging.info(f"theta: {theta}, chi: {chi}, gamma: {gamma}")

omega = np.array([np.cos(chi) * np.sin(theta), np.sin(chi) * np.sin(theta), np.cos(theta)])

# PyVista plotter
plotter = pv.Plotter(shape=(1, 1))

# plot Volume
sphere = pv.Sphere(radius=2, center=B_origin)
plotter.add_mesh(sphere, color="white", show_edges=False, opacity=1)

# plot Sun
Rsun = 696340 / 1e3  # km
sun = pv.Sphere(radius=Rsun, center=origin - np.array([0, 0, Rsun]), theta_resolution=100, phi_resolution=100)
plotter.add_mesh(sun, color="#ffbf00", show_edges=False, opacity=1)

# Vector B
arrow = pv.Arrow(
    start=B_origin, direction=B * vector_length, tip_length=0.1, tip_radius=0.02, shaft_radius=0.01, scale="auto"
)
plotter.add_mesh(arrow, color="blue")

# Vector omega
arrow = pv.Arrow(
    start=B_origin, direction=omega * vector_length, tip_length=0.1, tip_radius=0.02, shaft_radius=0.01, scale="auto"
)
plotter.add_mesh(arrow, color="black")

# draw line from Bz to origin
line = pv.Line(pointa=2 * B_origin, pointb=origin)
plotter.add_mesh(line, color="black", line_width=1)

plotter.set_focus(B_origin)


# =========

logging_config.init(logging.INFO)

term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()
nu = np.arange(reference_nu_sm1 - 1e11, reference_nu_sm1 + 1e11, 1e8)  # Hz

atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=B_norm, delta_v_thermal_cm_sm1=5_000_00)
radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_w(h_arcsec=0.725 * height)

# Rotate the radiation tensor from the Sun to the B field
D = WignerD(alpha=chi_B, beta=theta_B, gamma=0, K_max=2)
J_B = rotate_J(J=radiation_tensor, D=D)

atom = TwoTermAtom(
    term_registry=term_registry,
    transition_registry=transition_registry,
    atmosphere_parameters=atmosphere_parameters,
    radiation_tensor=J_B,
    disable_r_s=True,
    disable_n=True,
)

atom.add_all_equations()
rho = atom.get_solution_direct()

radiative_transfer_coefficients = RadiativeTransferCoefficients(
    atmosphere_parameters=atmosphere_parameters,
    transition_registry=transition_registry,
    nu=nu,
    chi=chi,
    theta=theta,
    gamma=gamma,
    chi_B=chi_B,
    theta_B=theta_B,
)

eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
eta_sQ = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=1)
eta_sU = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=2)
eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)
norm = np.max(np.abs(eta_sI))
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)

ax[0].plot(nu, eta_sI / norm, "g-", label=r"$\eta_s^I$")
ax[1].plot(nu, eta_sQ / norm, "r-", label=r"$\eta_s^Q$")
ax[2].plot(nu, eta_sU / norm, "y-", label=r"$\eta_s^U$")
ax[3].plot(nu, eta_sV / norm, "b-", label=r"$\eta_s^V$")

plt.xlabel("Frequency (Hz)")
for i in range(4):
    ax[i].set_ylim(-1.05, 1.05)
    ax[i].grid()
    ax[i].legend()
plotter.show_axes()
plotter.show(interactive_update=True)
plt.show()

plotter.close()
