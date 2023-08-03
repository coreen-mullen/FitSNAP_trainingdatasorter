import os, time
import numpy as np 
import pandas as pd  
import seaborn as sns

"""
Python script for performing a fit and immediately calculating test errors after the fit.

Test errors are reported for MAE energy (eV/atom) and MAE force (eV/Angstrom), if using LAMMPS 
metal units.

Serial:

    python example.py

Parallel:

    mpirun -np 2 python example.py

NOTE: See below for info on which variables to change for different options.
"""

from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap

# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD

# Create an input dictionary containing settings.
settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 0
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "."
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_BCC" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08",
    "Elastic_BCC"   :  "1.0    0.0     1.00E-08        1.00E-08        0.0001",
    "Elastic_FCC"   :  "1.0    0.0     1.00E-09        1.00E-09        1.00E-09",
    "GSF_110"       :  "1.0    0.0      100             1               1.00E-08",
    "GSF_112"       :  "1.0    0.0      100             1               1.00E-08",
    "Liquid"        :  "1.0    0.0       4.67E+02        1               1.00E-08",
    "Surface"       :  "1.0    0.0       100             1               1.00E-08",
    "Volume_A15"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_BCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_FCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Alternatively, settings could be provided in a traditional input file:
# settings = "../../Ta_Linear_JCP2014/Ta-example.in"
    
# Create a FitSnap instance using the communicator and settings:
fitsnap = FitSnap(settings, comm=comm, arglist=["--overwrite"])
conf = fitsnap.config.sections

################### MAKE FITSNAP FITS FROM TRAINING DATA SORTER ###################
# Instead of using the [GROUPS] section in the input file or dictionary,
# read groups from CSV files

# Designate a training CSV to get groups and training file sources
training_csv_path = "./fitsnap_csvs_for_TaLinear_31_bins"
group_chunk_csvs = [f for f in os.listdir(training_csv_path) if f.endswith(".csv")]
# Sort so that energies are always in ascending order
sort_key = lambda x: int(x[x.rfind("chunk")+5:].replace(".csv",""))
group_chunk_csvs = sorted(group_chunk_csvs, key = sort_key)
group_weight_dict = {'training_size': 1.0, 'testing_size': 0.0, 'eweight': 1, 'fweight': 100, 'vweight': 1e-12}

# Initiate new FitSNAP JSON scraper (see function above)
scr = fitsnap.scraper
scrargs = conf["SCRAPER"].scraper, fitsnap.pt, fitsnap.config
scr.__init__(*scrargs)
scr.group_table = {}
scr.tests = {}

# Overwrite original FitSNAP groups from settings
gt0 = conf["GROUPS"].group_table
print("DEBUG: original group table, dtype: ")
print(gt0)
print(type(gt0))

# Create new group table
# use constant settings for energy, force, and stress weights (see 'group_weight_dict' above)
new_gt = {}
print("Extracting group data from chunk CSVs")
print("\tgroup name : number of JSONs")
for gc in group_chunk_csvs:
    csv_fullpath = f"{training_csv_path}/{gc}"
    df = pd.read_csv(csv_fullpath)
    group_name = gc.replace(".csv","")
    new_gt[group_name] = group_weight_dict # constant values for all groups
    group_jsons = df["file"].tolist()
    scr.files[group_name] = group_jsons
    print(f"\t{group_name}: {len(group_jsons)}")

conf["GROUPS"].group_table = new_gt
print("DEBUG: overwritten FitSNAP group table")
print(conf["GROUPS"].group_table)

print("UNFINISHED SCRIPT! next step: do fitsnap fits with sorted training data")
exit()

# Scrape configurations to create and populate the `snap.data` list of dictionaries with structural info.
fitsnap.scrape_configs()
# Calculate descriptors for all structures in the `snap.data` list.
# This is performed in parallel over all processors in `comm`.
# Descriptor data is stored in the shared arrays.
fitsnap.process_configs()
# Now we can access the A matrix of descriptors:
# print(fitsnap.pt.shared_arrays['a'].array)
# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
fitsnap.pt.all_barrier()
# Perform a fit using data in the shared arrays.
fitsnap.perform_fit()
# Can also access the fitsnap dataframe here:
# print(snap.solver.df)
# WriteLAMMPS potential files and error analysis.
fitsnap.write_output()
