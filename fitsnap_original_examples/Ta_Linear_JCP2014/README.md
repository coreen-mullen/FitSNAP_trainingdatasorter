## FitSnap3 Tantalum example

This example will generate a potential for tantalum as published in 
Thompson, A.P. et. al, J. Comp. Phys. 285 (2015) 316-330.  This version 
of the tantalum potential uses the linear version of SNAP.

#### Running this example:

To run this example, use the following command in this directory:

    python -m fitsnap3 Ta-example.in

#### Files in this Directory

`Ta-example.in`

Input file containing parameters to run FitSNAP and generate
the tantalum potential

`JSON/`

Directory that contains all the training configurations which are organized
into different groups.

#### Files generated by example:

`Ta_pot.snapcoeff`

SNAP potential coefficient file that contains all the beta coefficients for 
this potential.  This is one of two files needed to use this potential in LAMMPS

`Ta_pot.snapparam`

SNAP potential parameters file that contains the hyperparameters and options used during 
the fit for this potential.  This is one of two files needed to use this potential in LAMMPS

`Ta_metrics.md`

Contains a variety of error metrics for all the training groups for this fit.

Note that the `20May21_Standard/` directory contains sample output for this example

`FitSNAP.df`

Pickled pandas dataframe whose columns contain various quantities associated with the fit. For those unfamiliar with pandas dataframes, this file can be loaded in python with:

        import pandas as pd
        data = pd.read_pickle("FitSNAP.df")

and we can extract data using typical pandas dataframe attributes, for example:

        print(data.columns) # see which columns we have
        print(data["Row_Type"].values) # print the values of row types in the A matrix

#### Important input parameters for this example

rcutfac = 4.67637  : Radial cutoff (hyperparameters) chosen for this potential
wj1 = 1.0 : Elemental weight on tantalum for density expansion
radelem1 = 0.5 : Tantalum per-element cutoff 
type1 = Ta : Chemical symbol for element which should match training files in JSON
quadraticflag = 0 : Quadratic SNAP is turned off, using linear SNAP

See docs/TEMPLATE.in for further information on input parameters

#### Tantalum data from:

The JSON configurations and hyperparameters used for this example are published in:

Thompson, A. P., Swiler, L. P., Trott, C. R., Foiles, S. M., & Tucker, G. J. (2015). 
Spectral neighbor analysis method for automated generation of quantum-accurate interatomic 
potentials. Journal of Computational Physics, 285, 316-330

**Note to Developers: Make sure this example still reproduces the same results when modifying code**

After running the example, use `python compare_snapcoeff.py` to calculate the max absolute difference in SNAP coefficients from the standard.
The values should agree within a near zero amount (machine precision or close). 
