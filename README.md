# IMERGE-FEP
This repository contains the code and files that form the basis of the manuscript "IMERGE-FEP: Improving Relative Free Energy Calculation Convergence with Chemical Intermediates"

## Contents
rgroupinterm - a tool for pairwise R-group enumeration and pruning for the automatic creation of intermediate molecules
rhfe_gromacs - the code, input files and md files to run the free energy perturbations described in the manuscript
rhfe_analysis - code for analysing FEP output
data - raw data used as input
supplemental information - sdf files & visualization of molecules from the perturbations that were run

## Installation
use setup.py for creating the environment for the intermediate generation
to run the rhfe gromacs scripts also install pmx, parmed, openff and openmm

## Generating intermediates for a molecular pair
The functionality and use of the intermediate generator is shown the jupyter notebook rgroup_enumeration.ipynb