Getting Started
===============

The recommended first run is the start-here demo:

`_demos/start_here/demo_basic_stokes_profile_synthesis.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/start_here/demo_basic_stokes_profile_synthesis.py>`_

The demo synthesizes the He I D3 Stokes profiles with a built-in multi-term atom in a constant-property slab. It constructs a frequency grid around the line, defines the line-of-sight and magnetic-field geometry, fills a prescribed anisotropic radiation tensor from the Allen radiation-field machinery, propagates an initially zero Stokes vector through the slab, and plots the emergent Stokes profiles.

To run it, install SolRaT and execute the script from a local copy of the repository. The demo can then be adapted to another line or setup by changing the model, atmosphere parameters, frequency grid, geometry, or radiation tensor.

The repository contains additional demos in `_demos/ <https://github.com/yakovkinii/SolRaT/tree/master/_demos>`_. These demos are also used as runnability checks, so they are the best source of short, current examples.
