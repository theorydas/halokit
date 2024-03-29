# halokit

A collection of code, functions and units that provide utility in my dark matter halo reasearch ranging from calculating orbit quantities to updating density distributions due to custom self annihilations. Unless stated otherwise in the list below, everything has been coded and maintented by myself.

## Contents

The package is split into files depending on their usage theme or origin.

* `units.py` - All relevant unit conversions and constants.
* `basic.py` - Basic astrophysical and gravitational wave related functions and general utilities used everywhere.
* `halos.py` - Related to the creation and properties of CDM halos.
* `sadm.py` - Self annihilating dark matter density profiles and code to isotropically evolve these systems.
* `dephasing.py` - Functions that produce the fiducial static CDM dephasing and empirical approximation for dynamic CDM systems.
* `phase.py` - Utilities to calculate phase and dephasing related quantities.
* `waveforms.py` - Waveform and mismatch builders from dephasing and other relevant quantities.
* `evolution.py` - Code and differentials for evolving the IMRI-Halo systems while to account for feedback.
* `dfriction.py` - Energy-related quantities for the IMRI-Halo systems.

* `HaloFeedback.py` - A stable iteration of Bradley's HaloFeedback package: https://github.com/bradkav/HaloFeedback altered to support fully customisable spikes.
* `gravatoms.py` - Ionisation rate calculations for gravitational atom systems, mostly coded by Gimmy.

## Requirements

The code heavily realies on `numpy` and `scipy` for most of its calculations and requires `tqdm`.
Currently gravatoms.py requires `sympy` and `mpmath` to operate.