<H1>SolRaT</H1>

SolRaT (Solar Radiative Transfer) is a forward modeling code for non-LTE (and optionally LTE) transfer of 
radiation in stellar atmospheres.

The code implements the multi-term atom model, described in 
`Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)`. 
SolRaT supports atomic level polarization, arbitrary magnetic fields (intermediate Paschen-Back effect), 
Hanle effect and many other features.

The code is written in python, currently tested on Windows and Ubuntu Linux. 
The code is expected to work on all systems that fully support python 3.11. 
SolRaT is currently in beta testing. Journal article and detailed documentation are pending.
Until then, if SolRaT has found use in your research, please cite it as 
```
Yakovkin I. I. SolRaT (2023) [computer software]. Retrieved from https://www.yakovkinii.com/solrat/
```

How to run:
```bash
git clone https://github.com/yakovkinii/SolRaT.git
pip install -r requirements.txt
python ./run_all_tests.py
```

Some examples of how to use SolRaT are available in the `_demos` directory.

Keywords:
Non-LTE, Stokes Profiles, Inversion, Synthesis, Paschen-Back, Hanle, Zeeman, 
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Polarization, 
Spectral Lines, Two-Term Atom Model

Copyright (2023) Ivan I. Yakovkin
