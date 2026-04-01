<H1>SolRaT</H1>

[![Documentation](https://img.shields.io/badge/read-TheDocs-eee?logoColor=black)](https://solrat.readthedocs.io/latest/)
[![Homepage](https://img.shields.io/badge/homepage-solrat-000000?logoColor=white)](https://www.yakovkinii.com/solrat/)
![License](https://img.shields.io/badge/license-MIT-00ff00)
[![PyPi Version](https://img.shields.io/pypi/v/solrat)](https://pypi.org/project/solrat)
![Language](https://img.shields.io/badge/language-Python-3776AB?logoColor=white)
![Supported Platforms](https://img.shields.io/badge/platform-any-ffffff?logoColor=black)
![Manual Coverage Check](https://img.shields.io/badge/coverage-99%25-0000ff)
[![Coverage Status](https://coveralls.io/repos/github/yakovkinii/SolRaT/badge.svg?branch=master)](https://coveralls.io/github/yakovkinii/SolRaT?branch=master)

SolRaT (Solar Radiative Transfer) is a flexible forward modeling code for non-LTE transfer of 
radiation in stellar atmospheres.

The code provides a three-level framework for modeling radiative transfer:
1. **Public API** allows to run built-in RT models. Currently, SolRaT ships the non-LTE multi-term atom model ([LL04](#References)) with multiple constant-slab atmosphere stratification.  
2. **Modeling API** allows to extend existing models or create completely new ones.    
3. **SolRaT Engine** introduces a meta-language that allows the user to write a human-readable code that directly 
resembles the underlying mathematical equations. The user does not need to focus on code optimization, 
as it is handled under the hood. 

#### Key Features:
- **Physics**: Solves non-LTE radiative transfer for multi-term atoms.
- **Magnetic Fields**: Handles arbitrary magnetic field strengths (Zeeman, Hanle, Paschen-Back effects).
- **Atmosphere**: Supports multi-slab atmospheres for height stratification under anisotropic illumination ([ATL08](#References), [HAZEL2](#References)).
- **Flexibility**: Provides a high-level meta-language allowing for writing code that directly resembles the underlying mathematical equations.
- **Accessibility**: SolRaT is free, open-source, and platform-independent.
- **Performance**: SolRaT uses high-performance libraries that leverage SIMD instruction sets.
- **Extensibility**: A clear Modeling API allows for quick prototyping and model adjustments for specific contexts.


#### How to run:
1. Install SolRaT from PyPi: ```pip install solrat```
2. Explore the [demos](https://github.com/yakovkinii/SolRaT/tree/master/_demos) and [tests](https://github.com/yakovkinii/SolRaT/tree/master/_tests) for usage examples.
3. Explore the [documentation](https://solrat.readthedocs.io/latest/).

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
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Polarization, 
Spectral Lines, Two-Term Atom Model

Copyright (2023) Ivan I. Yakovkin
