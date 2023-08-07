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
import os, time, copy, json
import numpy as np 
import pandas as pd  
import seaborn as sns
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.units.units import convert

# Designate a training CSV to get groups and training file sources
training_csv_path = "./fitsnap_csvs_for_TaLinear_31_bins"
new_potential_path = training_csv_path.replace("csvs", "fit")

# Set desired group weights (must be constant for now)
group_weight_dict = {'training_size': 1.0, 'testing_size': 0.0, 'eweight': 1, 'fweight': 100, 'vweight': 1e-12}

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
    "compute_testerrs": 0,
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

# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#################################################### FUNCTIONS
###### csv path band-aid for now
def swap_json_filepath(json_abs_path):
    cwd = os.getcwd()
    current_rootpath = "/home/cmmulle/FitSNAP/./examples"
    desired_rootpath = "./fitsnap_original_examples"
    new_path = json_abs_path.replace(current_rootpath, desired_rootpath)

    # print("DEBUG: json_abs_path: ", json_abs_path)
    # print("new_path: ", new_path)
    return new_path

########################## fitsnap custom scraper for reading from CSV
def scrape_configs(scr, json_group_list):
    scr.all_data = [] # Reset to empty list in case running scraper twice.
    scr.conversions = copy.copy(scr.default_conversions)
    data_path = scr.config.sections["PATH"].datapath
    for i, file_pair in enumerate(json_group_list):
        file_name, group_name = file_pair
        if file_name.endswith('.json'):
            with open(file_name) as file:
                if file.readline()[0]=="{":
                    file.seek(0)
                try:
                    scr.data = json.loads(file.read(), parse_constant=True)
                except Exception as e:
                    scr.pt.single_print(f"Trouble parsing training data: {file_name}")
                    scr.pt.single_print(f"{e}")
                assert len(scr.data) == 1, "More than one object (dataset) is in this file"
                scr.data = scr.data['Dataset']
                assert len(scr.data['Data']) == 1, "More than one configuration in this dataset"
                
                training_file = file_name.split("/")[-1]
                scr.data['File'] = training_file
                scr.data['Group'] = group_name
                # Move data up one level
                scr.data.update(scr.data.pop('Data')[0])  
                for key in scr.data:
                    if "Style" in key:
                        if key.replace("Style", "") in scr.conversions:
                            temp = scr.config.sections["SCRAPER"].properties[key.replace("Style", "")]
                            temp[1] = scr.data[key]
                            scr.conversions[key.replace("Style", "")] = convert(temp)
                for key in scr.config.sections["SCRAPER"].properties:
                    if key in scr.data:
                        scr.data[key] = np.asarray(scr.data[key])
                natoms = np.shape(scr.data["Positions"])[0]
                scr.data["QMLattice"] = (scr.data["Lattice"] * scr.conversions["Lattice"]).T

                # Populate with LAMMPS-normalized lattice
                del scr.data["Lattice"]  

                # TODO Check whether "Label" container useful to keep around
                if "Label" in scr.data:
                    del scr.data["Label"] 

                if not isinstance(scr.data["Energy"], float):
                    scr.data["Energy"] = float(scr.data["Energy"])

                # Insert electronegativities, which are per-atom scalars
                if (scr.config.sections["CALCULATOR"].per_atom_scalar):
                    if not isinstance(scr.data["Chis"], float):
                        scr.data["Chis"] = scr.data["Chis"]

                # Currently, ESHIFT should be in units of your training data (note there is no conversion)
                if hasattr(scr.config.sections["ESHIFT"], 'eshift'):
                    for atom in scr.data["AtomTypes"]:
                        scr.data["Energy"] += scr.config.sections["ESHIFT"].eshift[atom]
                scr.data["test_bool"] = scr.test_bool[i]
                scr.data["Energy"] *= scr.conversions["Energy"]
                scr._rotate_coords()
                scr._translate_coords()

                # TODO
                scr._weighting(natoms)

                scr.all_data.append(scr.data)
        else:
            scr.pt.single_print("Non-json file found: ", file_name)    

    return scr.all_data

####################################################################### train potentials
# Create a FitSnap instance using the communicator and settings:
fitsnap = FitSnap(settings, comm=comm, arglist=["--overwrite"])
conf = fitsnap.config.sections

# Instead of using the [GROUPS] section in the input file or dictionary,
# read groups from CSV files
group_chunk_csvs = [f for f in os.listdir(training_csv_path) if f.endswith(".csv")]
# Sort so that energies are always in ascending order
sort_key = lambda x: int(x[x.rfind("chunk")+5:].replace(".csv",""))
group_chunk_csvs = sorted(group_chunk_csvs, key = sort_key)

# Initiate new FitSNAP JSON scraper (see function above)
scr = fitsnap.scraper
scrargs = conf["SCRAPER"].scraper, fitsnap.pt, fitsnap.config
scr.__init__(*scrargs)
scr.group_table = {}
scr.tests = {}

# Overwrite original FitSNAP groups from settings
gt0 = conf["GROUPS"].group_table
# print("DEBUG: original group table, dtype: ")
# print(gt0)
# print(type(gt0))

# Create new group table
# use constant settings for energy, force, and stress weights (see 'group_weight_dict' above)
# note: remake chunk CSVs with local example

print("Extracting group data from chunk CSVs")
print("\tgroup name : number of JSONs")
for gc in group_chunk_csvs:
    csv_fullpath = f"{training_csv_path}/{gc}"
    df = pd.read_csv(csv_fullpath)
    group_name = gc.replace(".csv","")
    scr.group_table[group_name] = group_weight_dict
    group_jsons = df["file"].tolist()

    # test group JSON paths (for early script versions)
    test_json = group_jsons[0]
    if not os.path.exists(test_json):
        print("\tUpdating JSON pathing")
        group_jsons = [swap_json_filepath(j) for j in group_jsons]
        print("\tDEBUG: old first entry of group_jsons vs. new")
        print("\t",test_json)
        print("\t",group_jsons[0])

    scr.files[group_name] = group_jsons
    print(f"\t{group_name}: {len(group_jsons)}")
    scr.tests[group_name] = []
    njsons = len(group_jsons)
    scr.group_table[group_name]['training_size'] = njsons
    scr.group_table[group_name]['testing_size'] = 0

scr.configs = scr.files

# Make sure to also update the configuration file group table to avoid confusion
conf["GROUPS"].group_table = scr.group_table

# print("DEBUG: overwritten FitSNAP group table")
# print(scr.group_table)
# print(conf["GROUPS"].group_table)

# Before beginning fit, make sure output files get put into new directory
# Set up output directory
if not os.path.exists(new_potential_path):
    os.mkdir(new_potential_path)

# Overwrite original outfile settings with new directory
# NOTE: weirdly, section OUTFILE attributes do not match to input variable names!
# see print statement below
# print(dir(conf['OUTFILE']))
cwd = os.getcwd()
metrics0 = settings["OUTFILE"]["metrics"]
potential0 = settings["OUTFILE"]["potential"]
conf["OUTFILE"].metric_file = f"{new_potential_path}/{metrics0}"
conf["OUTFILE"].potential_name = f"{new_potential_path}/{potential0}"

# print("DEBUG: make sure configuration file has correct pathing")
# print(conf["OUTFILE"].metric_file)
# print(conf["OUTFILE"].potential_name)

# Scrape configurations to create and populate the `snap.data` list of dictionaries with structural info.
scr.divvy_up_configs() # TODO only use when mpi soft size > 1
fitsnap.data = scrape_configs(scr, scr.configs)

# Calculate descriptors for all structures in the `snap.data` list.
# This is performed in parallel over all processors in `comm` (if MPI enabled).
# Descriptor data is stored in the shared arrays.
fitsnap.process_configs()

# Now we can access the A matrix of descriptors:
# print(fitsnap.pt.shared_arrays['a'].array)

# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
fitsnap.pt.all_barrier()

# Perform a fit using data in the shared arrays.
fitsnap.perform_fit()

# Can also access the fitsnap dataframe here:
# NOTE: we could probably put a bunch of different fits in a for loop in this script and access the metrics_file information right here
# print(snap.solver.df)

# Write LAMMPS potential files and error analysis in new directory
fitsnap.write_output()


