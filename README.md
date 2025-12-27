<H1>Welcome to SolRaT</H1>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SolRaT (stands for "Solar Radiative Transfer") is a Python package 
for simulating the non-LTE transfer of solar radiation in the atmosphere.

The code implements the **multi-term atom model**, described in 
`Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)`. 
The multi-term model accounts for the following Non-LTE effects:

- Atomic level polarization
- Arbitrary magnetic fields (Linear Zeeman effect, Paschen-Back effect, and intermediate cases)
- Hanle and other effects

The Two-Term model is currently implemented for synthesis only. 
The implementation of Stokes profile inversion is pending.

The code is written in Python for Windows, Linux and MacOS, currently tested on Windows only.

---

SolRaT is designed to facilitate the work with complicated equations by creating high-level abstractions.
It accomplishes the following:
- Make the implementation resemble the underlying mathematical expressions
- Automatic performance optimization, including SIMD vectorization, partial evaluation, and more

For example, consider the following expression:

$`\sum_{J_l, M_l, J_u, M_u} A(J_l, M_l, J_u, M_u)`$

In most RTE codes it is implemented in the following manner:
```python
result = 0
for j_l in range(abs(l_l - s), l_l + s + 1):
    for m_l in range(-j_l, j_l + 1):
        for j_u range(abs(l_u - s), l_u + s + 1):
            for m_u in range(-j_u, j_u + 1):
                result += A(j_l, m_l, j_u, m_u)
return result
```
The optimization of such code (e.g. explicitly accounting for additional constraints due to 3J/6J/9J symbols) makes it much more complicated, and the codebase becomes hard to maintain and modify.

Meanwhile, SolRaT allows to write the code in a way that resembles the original equations, separating the physics from the optimization, with the latter occurring elsewhere:
```python
return summate(
    lambda Jl, Ml, Ju, Mu: A(Jl, Ml, Ju, Mu),
    Jl=TRIANGULAR(L, S),
    Ml=PROJECTION("J"),
    Ju=TRIANGULAR(L, S),
    Mu=PROJECTION("Ju"),
)
```
Under significant optimization, the implementation of equations in SolRaT remains unchanged, with 1-to-1 correspondence to the original equations.

This allows users to modify SolRaT code to account for different approximations and approaches with confidence.

Keywords:
Non-LTE, Stokes Profiles, Inversion, Synthesis, Paschen-Back, Hanle, Zeeman, 
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Polarization, 
Spectral Lines, Two-Term Atom Model
