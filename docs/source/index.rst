SolRaT Documentation
====================

SolRaT (Solar Radiative Transfer) is a flexible, forward-modeling, non-LTE radiative transfer code for stellar atmospheres. It implements the statistical equilibrium and radiative transfer equations within the multi-term atom model, enabling detailed synthesis of Stokes profiles in magnetic fields of arbitrary strength, from the Zeeman to the Paschen-Back regimes.

.. image:: https://www.yakovkinii.com/solrat/media/solrat7.png
   :width: 600
   :alt: Model of He I D3 emission in a limb event under different magnetic fields
   :align: center

Key Features
------------
*   **Physics**: Solves non-LTE radiative transfer for multi-term atoms.
*   **Magnetic Fields**: Handles arbitrary magnetic field strengths (Zeeman, Hanle, Paschen-Back effects).
*   **Atmosphere**: Supports multi-slab atmospheres for height stratification.
*   **Flexibility**: Provides a high-level meta-language allowing you to write code that directly resembles the underlying mathematical equations.
*   **Accessibility** SolRaT is free, open-source, and platform-independent.
*   **Performance**: SolRaT uses high-performance libraries that leverage SIMD instruction sets.
*   **Extensibility**: A clear Modeling API allows for quick prototyping and model adjustments for specific contexts.


The figure above demonstrates a sample output: the modeling of He I D3 emission under varying magnetic field strengths.

The code provides a multi-level framework for modeling radiative transfer:

*   **Public API** allows to run built-in RT models. Currently, SolRaT ships the non-LTE multi-term atom model (LL04) with multiple constant-slab atmosphere stratification.
*   **Modeling API** allows to extend existing models or create completely new ones.
*   **SolRaT Engine** introduces a meta-language that allows the user to write a human-readable code that directly resembles the underlying the mathematical equations. The user does not need to focus on code optimization, as it is handled under the hood.

[LL04] Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)

How to Cite
-----------
If SolRaT contributes to your research, please cite it as:

    Yakovkin I. I. SolRaT (2023) [computer software]. Retrieved from https://www.yakovkinii.com/solrat/

Installation
------------
Install the latest release from PyPI:

.. code-block:: bash

   pip install solrat

For the development version, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/yakovkinii/SolRaT.git
   cd SolRaT
   pip install -e .

Next Steps
----------
Check out the :doc:`quickstart` guide for basic usage examples, or dive straight into the :doc:`api` reference.

.. toctree::
   :maxdepth: 2
   :caption: QuickStart:

   quickstart

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Api Reference:

   api