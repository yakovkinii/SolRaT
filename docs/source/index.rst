SolRaT Documentation
====================

SolRaT (Solar Radiative Transfer) is an open-source forward-synthesis code for polarized spectral lines in magnetized stellar atmospheres, with interchangeable multi-term and multi-level atomic descriptions, prescribed or self-consistent radiation tensors, and reusable atmosphere models.

.. toctree::
   :hidden:
   :maxdepth: 1

   Home <self>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: INTRODUCTION

   about
   installing
   getting_started

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: ATMOSPHERE MODELS

   constant_property_slab
   multi_slab_synthesis
   prescribed_jkq_stratified
   self_consistent_nlte
   mt_ml_comparison

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: EXTENDING SOLRAT

   adding_spectral_lines
   modifying_see
   modifying_rte
   getting_help

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API REFERENCE

   api

.. raw:: html

   <div class="solrat-link-bar">
     <a href="https://www.yakovkinii.com/solrat/">Project website</a>
     <a href="https://github.com/yakovkinii/SolRaT/">GitHub repository</a>
     <a href="https://arxiv.org/abs/2609.32850">arXiv article</a>
     <a href="https://pypi.org/project/solrat/">PyPI package</a>
   </div>

   <section class="solrat-card-section">
     <h2>Introduction</h2>
     <div class="solrat-card-grid">
       <a class="solrat-card" href="about.html">
         <span class="solrat-card-title">About SolRaT</span>
         <span class="solrat-card-text">Scope, physical assumptions, features, and reference article.</span>
       </a>
       <a class="solrat-card" href="installing.html">
         <span class="solrat-card-title">Installing</span>
         <span class="solrat-card-text">PyPI installation, editable development installs, and repository setup.</span>
       </a>
       <a class="solrat-card" href="getting_started.html">
         <span class="solrat-card-title">Getting Started</span>
         <span class="solrat-card-text">Where to begin, which demo to copy, and what the first synthesis does.</span>
       </a>
     </div>
   </section>

   <section class="solrat-card-section">
     <h2>Atmosphere Models</h2>
     <div class="solrat-card-grid">
       <a class="solrat-card" href="constant_property_slab.html">
         <span class="solrat-card-title">Constant-Property Slab Synthesis</span>
         <span class="solrat-card-text">A single homogeneous slab under prescribed illumination.</span>
       </a>
       <a class="solrat-card" href="multi_slab_synthesis.html">
         <span class="solrat-card-title">Multi-Slab Synthesis</span>
         <span class="solrat-card-text">Several slabs chained along the line of sight.</span>
       </a>
       <a class="solrat-card" href="prescribed_jkq_stratified.html">
         <span class="solrat-card-title">Prescribed-J<sup>K</sup><sub>Q</sub> Stratified Atmosphere</span>
         <span class="solrat-card-text">Height-dependent atmospheric parameters with an externally supplied radiation tensor.</span>
       </a>
       <a class="solrat-card" href="self_consistent_nlte.html">
         <span class="solrat-card-title">Self-Consistent NLTE Atmosphere</span>
         <span class="solrat-card-text">Coupled statistical-equilibrium and transfer iteration in a stratified atmosphere.</span>
       </a>
       <a class="solrat-card" href="mt_ml_comparison.html">
         <span class="solrat-card-title">Multi-Term vs Multi-Level Comparison</span>
         <span class="solrat-card-text">How the same atmospheric setup can be used with different atomic descriptions.</span>
       </a>
     </div>
   </section>

   <section class="solrat-card-section">
     <h2>Extending SolRaT</h2>
     <div class="solrat-card-grid">
       <a class="solrat-card" href="adding_spectral_lines.html">
         <span class="solrat-card-title">Adding More Spectral Lines</span>
         <span class="solrat-card-text">How to add atomic data and combine component transition probabilities.</span>
       </a>
       <a class="solrat-card" href="modifying_see.html">
         <span class="solrat-card-title">Modifying SEE</span>
         <span class="solrat-card-text">Where statistical-equilibrium rates are implemented and how caching enters.</span>
       </a>
       <a class="solrat-card" href="modifying_rte.html">
         <span class="solrat-card-title">Modifying RTE</span>
         <span class="solrat-card-text">Where transfer coefficients are implemented and how to extend their summations.</span>
       </a>
       <a class="solrat-card" href="getting_help.html">
         <span class="solrat-card-title">Getting Help</span>
         <span class="solrat-card-text">Where to ask questions about assumptions, use cases, and extensions.</span>
       </a>
     </div>
   </section>

   <section class="solrat-card-section">
     <h2>API Reference</h2>
     <div class="solrat-card-grid">
       <a class="solrat-card" href="api.html#built-in-models">
         <span class="solrat-card-title">Built-in Models</span>
         <span class="solrat-card-text">Preconfigured model registry and public model entry points.</span>
       </a>
       <a class="solrat-card" href="api.html#base-atom-model">
         <span class="solrat-card-title">Base Atom Model</span>
         <span class="solrat-card-text">Abstract containers and interfaces shared by atomic descriptions.</span>
       </a>
       <a class="solrat-card" href="api.html#multi-level-atom-model">
         <span class="solrat-card-title">Multi-Level Atom Model</span>
         <span class="solrat-card-text">Multi-level statistical-equilibrium and transfer implementation.</span>
       </a>
       <a class="solrat-card" href="api.html#multi-level-atom-model-lte">
         <span class="solrat-card-title">Multi-Level Atom Model (LTE)</span>
         <span class="solrat-card-text">LTE specialization of the multi-level description.</span>
       </a>
       <a class="solrat-card" href="api.html#multi-term-atom-model">
         <span class="solrat-card-title">Multi-Term Atom Model</span>
         <span class="solrat-card-text">Multi-term statistical-equilibrium, transfer, and Paschen-Back machinery.</span>
       </a>
       <a class="solrat-card" href="api.html#multi-term-atom-model-legacy">
         <span class="solrat-card-title">Multi-Term Atom Model (legacy)</span>
         <span class="solrat-card-text">Legacy multi-term implementation retained for reference.</span>
       </a>
       <a class="solrat-card" href="api.html#multi-term-atom-model-lte">
         <span class="solrat-card-title">Multi-Term Atom Model (LTE)</span>
         <span class="solrat-card-text">LTE specialization of the multi-term description.</span>
       </a>
       <a class="solrat-card" href="api.html#shared-public-api">
         <span class="solrat-card-title">Shared Public API</span>
         <span class="solrat-card-text">Atmospheres, geometry, Stokes objects, profiles, and utilities.</span>
       </a>
       <a class="solrat-card" href="api.html#solrat-engine">
         <span class="solrat-card-title">SolRaT Engine</span>
         <span class="solrat-card-text">Summation, reduction, and compiled-operator internals.</span>
       </a>
     </div>
   </section>
