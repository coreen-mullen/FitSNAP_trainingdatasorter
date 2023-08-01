import sys, statistics, os, json, operator, glob
from glob import glob
import pathlib 
from pathlib import Path
import os.path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import shutil 
import csv
# %matplotlib inline

# ###########################
file_search_term = ".json"
dir_search_term = "JSON"
element_search_terms = ["Ta"] # ['WBe'] # ["InP"]
model_search_terms = ["Linear"] # ["PRB2019"] # ["JPCA2020"]
# ##################
if len(model_search_terms) != len(element_search_terms):
    print("im elegantly crashing cause these two vars need to be the same length.")

csv_columns_str = "element,model,search_item,rel_path,file,Group\n"
# ################
def pull(json_files):
    temp_item1, temp_item2 = 0, 0
    with open(json_files) as file:
        txt = file.readlines()
        if len(txt) == 1:
            json_data =json.loads(txt[0])
        else:
            json_data = json.loads(txt[1])
        ## (she will show you later why)
        for items in json_data["Dataset"]["Data"]:
             temp_item1 = items["Energy"]
             temp_item2 = items["NumAtoms"]
    return temp_item1, temp_item2
  
# ################
collect_str_rows_for_a_csv = []
cwd = os.getcwd() 
for current_dir, sub_dirs, files in os.walk("."):
    if "example_walk" in current_dir:
        continue
    for model in model_search_terms:
        for element in element_search_terms:
            if dir_search_term in current_dir and element in current_dir and model in current_dir:
                json_files = [f"{cwd}/{current_dir}/{f}" for f in files if file_search_term in f]
                for f in json_files:
                    need, nitems = pull(f)
                    normalized_need = need/nitems
                    #print(f, nitems, need, normalized_need)
                    just_group = f[f.rfind("/")+1:] 
                    relpath_json = current_dir[current_dir.rfind("/")+1:]
                    relpath_group = relpath_json[relpath_json.rfind("/")+1:]
                    row_str = f"{element},{model},{need},{current_dir},{f},{relpath_group}\n"
                    collect_str_rows_for_a_csv.append(row_str)    

if collect_str_rows_for_a_csv == []:
    print("!!!! The list is empty and will need to be fixed.")
    exit()
            
# ##############################
csv_name = f'{element}{model}Test2.csv'

with open(csv_name, "w") as lo:
   lo.write(csv_columns_str)
   for row in collect_str_rows_for_a_csv:
        lo.write(row) 
         
# #####################################
data = pd.read_csv(csv_name)

D = data.sort_values("search_item").reset_index(drop=True)
desired_groups = 10
num_ebins = len(D)//desired_groups
num_rows = D.shape[0]
chunks = [D[i:i+num_ebins] for i in range(0,len(D), num_ebins)]
file_path_name = f'fitsnap_csvs_for_{element}{model}_{desired_groups}'
# ###############################################
for i, chunk in enumerate(chunks[:]):
    folder_name = 2
    chunk_num = i+1
    if not os.path.exists(f'./fitsnap_csvs_for_{element}{model}_{desired_groups}'):
         os.mkdir(f"./fitsnap_csvs_for_{element}{model}_{desired_groups}")
    csv_full_name = f'./{file_path_name}/{element}{model}_chunk{chunk_num}.csv'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    chunk.to_csv(csv_full_name,index=False)
