<H1>Welcome to SolRaT</H1>
SolRaT (stands for "Solar Radiative Transfer") is a Python package 
for simulating the non-LTE transfer of solar radiation in the atmosphere.

The project consists of 2 parts: the core and the Two-Term atom model.

The **core**, designed to facilitate the work with complicated equations
by creating high-level abstractions.
It makes the implementation resemble the underlying mathematical expressions.

The **Two-Term model** is an implementation of the Two-Term atom model, 
described in 
`Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)`. 
The Two-Term model accounts for the following Non-LTE effects:

- Atomic level polarization
- Arbitrary magnetic fields (Linear Zeeman effect, Paschen-Back effect, and intermediate cases)
- Hanle and other effects

The Two-Term model is currently implemented for synthesis only. 
The implementation of Stokes profile inversion is pending.

The code is written in Python for Windows and Linux platforms.

Keywords:
Non-LTE, Stokes Profiles, Inversion, Synthesis, Paschen-Back, Hanle, Zeeman, 
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Polarization, 
Spectral Lines, Two-Term Atom Model