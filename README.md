<H1>SolRaT</H1>

[![](https://img.shields.io/badge/read-TheDocs-eee?logoColor=black)](https://solrat.readthedocs.io/latest/)
[![](https://img.shields.io/badge/homepage-solrat-000000?logoColor=white)](https://www.yakovkinii.com/solrat/)
![](https://img.shields.io/badge/language-Python-3776AB?logoColor=white)
![](https://img.shields.io/github/license/henriquesebastiao/badges?style=flat&color=22c55e)
[![](https://img.shields.io/pypi/v/solrat)](https://pypi.org/project/solrat)
![](https://img.shields.io/badge/platform-Linux-ffffff?logoColor=black)
![](https://img.shields.io/badge/platform-macOS-000000?logoColor=white)
![](https://img.shields.io/badge/platform-Windows-0078D6?logoColor=white)
![](https://img.shields.io/badge/coverage-90%25-0000ff)

SolRaT (Solar Radiative Transfer) is a flexible forward modeling code for non-LTE transfer of 
radiation in stellar atmospheres.

The code provides a multi-level framework for modeling radiative transfer:
1. **Public API** allows to run built-in RT models. Currently, SolRaT ships the non-LTE multi-term atom model ([LL04](#References)) with multiple constant-slab atmosphere stratification.  
2. **Modeling API** allows to extend existing models or create completely new ones.    
3. **SolRaT Engine** introduces a meta-language that allows the user to write a human-readable code that directly 
resembles the underlying the mathematical equations. The user does not need to focus on code optimization, 
as it is handled under the hood. 

#### Key highlights of SolRaT:
- Non-LTE radiative transfer
- Multi-term atom model allows for arbitrary magnetic fields
- Multi-slab atmosphere model allows to explicitly model the atmospheric stratification
- Flexible modeling API allows to adjust the physical model to specific problems
- Platform-independent python implementation
- Uses high-performance libraries (pandas, numpy) that leverage SIMD instruction sets


#### How to run:
1. Install SolRaT from PyPi: ```pip install solrat```
2. Explore the [demos](https://github.com/yakovkinii/SolRaT/tree/master/_demos) and [tests](https://github.com/yakovkinii/SolRaT/tree/master/_tests) for usage examples.
3. Explore the [documentation](https://solrat.readthedocs.io/latest/).

#### Citing:
SolRaT is currently in beta testing. Journal article is pending and the detailed documentation is under development.
In the meantime, if SolRaT has found use in your research, please cite it as 
```
Yakovkin I. I. SolRaT (2023) [computer software]. Retrieved from https://www.yakovkinii.com/solrat/
```

#### References:
[LL04] Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)

<h4>Keywords:</h4>
Non-LTE, Stokes Profiles, Inversion, Synthesis, Paschen-Back, Hanle, Zeeman, 
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Polarization, 
Spectral Lines, Two-Term Atom Model

Copyright (2023) Ivan I. Yakovkin
