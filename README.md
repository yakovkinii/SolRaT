# SolRaT 

[![Documentation](https://img.shields.io/badge/read-TheDocs-eee?logoColor=black)](https://solrat.readthedocs.io/latest/)
[![Homepage](https://img.shields.io/badge/homepage-solrat-000000?logoColor=white)](https://www.yakovkinii.com/solrat/)
![License](https://img.shields.io/badge/license-MIT-00ff00)
[![PyPi Version](https://img.shields.io/pypi/v/solrat)](https://pypi.org/project/solrat)
![Language](https://img.shields.io/badge/language-Python-3776AB?logoColor=white)
![Supported Platforms](https://img.shields.io/badge/platform-any-ffffff?logoColor=black)
[![Coverage Status](https://coveralls.io/repos/github/yakovkinii/SolRaT/badge.svg?branch=master)](https://coveralls.io/github/yakovkinii/SolRaT?branch=master)

SolRaT (Solar Radiative Transfer) is a flexible forward modeling code for non-LTE transfer of 
radiation in stellar atmospheres. SolRaT is specifically designed for prototyping and adjustments of the radiative transfer 
models for specific contexts.

#### Features available out-of-the-box:
- **RTE frameworks**: Multi-Term Atom; Multi-Level Atom; Multi-Term semi-LTE Atom [[LL04](#References)].
- **Magnetic Fields**: Zeeman &ndash; Hanle &ndash; Paschen-Back for Multi-Term Atom; 
Zeeman &ndash; Hanle for Multi-Level atom.
- **Atomic Level Polarization**: Fully supported.
- **Radiation Tensor $J_Q^K$**: Tabulated from [[ATL08](#References)] (for coronal/chromospheric lines) or NLTE self-consistent.
- **Atmospheres**: Multiple constant-property slabs.
- **Pre-configured lines**: He I D3, Mn I 5432.5 &Aring;, Ni I 5435.9 &Aring;, Fe I 5434.523 &Aring;.

#### Installation:

Install SolRaT directly from PyPi by running ```pip install solrat```.

#### Documentation:

Detailed documentation is available at [https://solrat.readthedocs.io/](https://solrat.readthedocs.io/latest/). 
Quick start example is available at [https://solrat.readthedocs.io/latest/quickstart.html](https://solrat.readthedocs.io/latest/quickstart.html).
Additional demos and validation against [[LL04](#References)] and [[HAZEL2](#References)] are available in [demos](https://github.com/yakovkinii/SolRaT/tree/master/_demos). 

#### Citing:
Journal article is pending. In the meantime, if SolRaT has found use in your research, please cite it as 
```
Yakovkin I. I. SolRaT (2023) [computer software]. Retrieved from https://www.yakovkinii.com/solrat/
```

#### References:
[LL04] Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)

[ATL08] Asensio Ramos, A., Trujillo Bueno, J., & Landi Degl’Innocenti, E. (2008). Advanced Forward Modeling and Inversion of Stokes Profiles Resulting from the Joint Action of the Hanle and Zeeman Effects. The Astrophysical Journal, 683(1), 542–565.

[HAZEL2] [Link](https://github.com/aasensio/hazel2)

<h4>Keywords:</h4>
Non-LTE, Stokes Profiles, Inversion, Synthesis, Paschen-Back, Hanle, Zeeman, 
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Spectral Line Polarization, 
Spectral Lines, Multi-Term Atom Model, Multi-Level Atom Model, Atomic Polarization. 

Copyright (2023) Ivan I. Yakovkin
