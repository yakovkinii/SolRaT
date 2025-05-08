import logging

import numpy as np
import pyvista as pv
import vtk
from yatools import logging_config

_vtk = vtk  # This is a hack for Windows to enable LaTeX rendering in PyVista

logging_config.init(logging.INFO)

# Vector B
Bx, By, Bz = 1.0, 2, 5
height = 2

# Origin and projection
origin = np.array([0.0, 0, 0])
B_origin = np.array([0.0, 0, height])
B_norm = np.linalg.norm([Bx, By, Bz])
B = np.array([Bx, By, Bz]) / B_norm
B_proj_xy = np.array([Bx, By, 0]) / B_norm
B_proj_z = np.array([0, 0, Bz]) / B_norm

# Compute angles
theta = np.arctan2(np.linalg.norm(B_proj_xy), np.linalg.norm(B_proj_z))
phi = np.arctan2(Bx, By)

# PyVista plotter
plotter = pv.Plotter()

# Add plane
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x) - 1e-5
# plane = pv.StructuredGrid(x, y, z)
# plotter.add_mesh(plane, color="orange", opacity=0.2, show_edges=False)

# plot sphere
sphere = pv.Sphere(radius=0.6, center=B_origin)
plotter.add_mesh(sphere, color="blue", show_edges=False, opacity=0.3)

# plot sphere
Rsun = 696340 / 10000  # km
sun = pv.Sphere(radius=Rsun, center=origin - np.array([0, 0, Rsun]))
plotter.add_mesh(sun, color="orange", show_edges=False, opacity=0.8)

Rearth = 6378 / 10000.0 * 10  # km
earth = pv.Sphere(
    radius=Rearth, center=origin - np.array([0, 151_000_000 / 10000 / 1.41, Rsun + 151_000_000 / 10000 / 1.41])
)
plotter.add_mesh(earth, color="green", show_edges=False, opacity=1)


# Vector B
arrow = pv.Arrow(start=B_origin, direction=B, tip_length=0.1, tip_radius=0.02, shaft_radius=0.01, scale="auto")
plotter.add_mesh(arrow, color="black")
plotter.add_point_labels([B_origin + B], ["B"], text_color="black", font_size=16, shape_opacity=0, show_points=False)

# draw line from B to Oz
line = pv.Line(pointa=B_origin + B, pointb=B_origin + B_proj_z)
plotter.add_mesh(line, color="black", line_width=1)

# draw line from Bz to origin
line = pv.Line(pointa=B_origin + B_proj_z, pointb=origin)
plotter.add_mesh(line, color="black", line_width=1)


# draw line from B to B_proj
line = pv.Line(pointa=B_origin + B, pointb=B_proj_xy)
plotter.add_mesh(line, color="black", line_width=1)

# Add projection of B onto XY plane
line = pv.Line(pointa=origin, pointb=B_proj_xy)
plotter.add_mesh(line, color="black", line_width=1)

# Add projection of B onto XY plane
line = pv.Line(pointa=B_origin, pointb=B_origin + B_proj_xy)
plotter.add_mesh(line, color="black", line_width=1)

# Add projection of Bxy onto X axis
line = pv.Line(pointa=origin, pointb=[B_proj_xy[0], 0, 0])
plotter.add_mesh(line, color="black", line_width=1)

# Add projection of Bxy onto Y axis
line = pv.Line(pointa=origin, pointb=[0, B_proj_xy[1], 0])
plotter.add_mesh(line, color="black", line_width=1)

# Add opposite lines
line = pv.Line(pointa=B_proj_xy, pointb=[B_proj_xy[0], 0, 0])
plotter.add_mesh(line, color="black", line_width=1)

line = pv.Line(pointa=B_proj_xy, pointb=[0, B_proj_xy[1], 0])
plotter.add_mesh(line, color="black", line_width=1)


# Arc for theta
arc_phi = pv.CircularArcFromNormal(
    center=B_origin, resolution=100, normal=np.cross(B_proj_z, B), polar=B_proj_z * 0.3, angle=np.degrees(theta)
)
plotter.add_mesh(arc_phi, color="purple")
midpoint_theta = B_origin + B_proj_z * 0.3 * np.cos(theta / 2) + B_proj_xy * 0.3 * np.sin(theta / 2)
plotter.add_point_labels(
    [midpoint_theta], [r"$\theta$"], text_color="purple", font_size=18, shape_opacity=0, show_points=False
)

# Add axes and labels
# plotter.add_axes()
# plotter.show_bounds(grid=None, location="outer", all_edges=True, bold=False, font_size=10, use_2d=False)
plotter.add_text("Vector B with Angles θ (azimuth) and φ (elevation)", font_size=12)


plotter.set_focus(
    origin - np.array([0, 151_000_000 / 10000 / 1.41, Rsun + 151_000_000 / 10000 / 1.41])
)  # Optional: center of rotation
# plotter.camera_position = 'xy'  # Optional: set view
# plotter.set_scale(xscale=1, yscale=1, zscale=1)  # Keep aspect ratio

# Force bounding box
# plotter.show_bounds(bounds=[-10, 10, -10, 10, -5, 10])

# set coordinate ranges
# plotter.view_xy()
# plotter.camera.SetPosition(0, 0, 2)
# plotter.enable_terrain_style()
# Show plot
# plotter.camera.SetParallelProjection(True)
plotter.show()
