import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cwd = os.getcwd()
collect_dirs = [d for d in os.listdir(cwd) if d.startswith("fitsnap_fit") and os.path.isdir(d)]
m_full_paths_2d = [[f"{cwd}/{collect_dir}/{f}" for f in os.listdir(collect_dir) if "metrics" in f] for collect_dir in collect_dirs]
m_full_paths = [item for items in m_full_paths_2d for item in items]
datacols = "model nbins eweight fweight label group weight_type train_or_test row_type ncount mae rmse rsq".split()
mcsv = "all_metrics.csv"

REDO_CSV = True
if not os.path.exists(mcsv) or REDO_CSV:
    all_data = []
    all_errs = []
    for i, m_full_path in enumerate(m_full_paths[:]):
        m_file = m_full_path[m_full_path.rfind("/")+1:]
        tmp = m_full_path.replace(m_file,"")[:-1]
        m_info_dir = tmp[tmp.rfind("/")+1:]
        info = m_info_dir.split("_")
        model, nbins, ef_info = info[3], info[4], info[6]
        eweight = float(ef_info[:ef_info.rfind("f")].replace("e",""))
        fweight = float(ef_info[ef_info.rfind("f"):].replace("f",""))
        label = f"{model}_{ef_info}"
        # print(label, nbins, eweight, fweight,)

        # parse metrics file 
        with open(m_full_path, "r") as f:
            mtxt = f.readlines()

        for row in mtxt:
            if "weight" not in row:
                continue
            r0 = row.split("|")
            r1, ncount, mae, rsme, rsq = [d.strip() for d in r0[1:6]]
            group, weight_type, train_or_test, row_type = [d.strip() for d in r1.replace("(","").replace(")","").replace("'","").replace("'","").lower().split(',')]
            data = [model, nbins, eweight, fweight, label, group, weight_type, train_or_test, row_type , float(ncount), float(mae), float(rsme), float(rsq)]
            # print(data)
            all_data.append(data)

    df0 = pd.DataFrame.from_records(all_data, columns=datacols)

    print(df0.shape)

    df0.to_csv(mcsv,index=False)
else:
    df0 = pd.read_csv(mcsv)

# df0 is "dataframe 0" and contains ALL the data from the metrics.md files

# for now, we only care about a small subset of the data, so use only group "*all" and "unweighted" rows
boolean_mask = (df0.group=="*all")&(df0.weight_type=="unweighted")&(df0.row_type!="stress")

df = df0.loc[boolean_mask,:]

## -------- example 1
fig, axes = plt.subplots(1,2, figsize=[7,3.5]) # default figsize is 6x4 inches
ax1, ax2 = axes
sns.scatterplot(data=df, x="eweight", y="mae", hue="row_type", ax=ax1)
sns.scatterplot(data=df, x="fweight", y="mae", hue="row_type", ax=ax2)
ax1.set(xscale="log")
ax2.set(xscale="log")

plt.tight_layout()
plt.savefig("mae_scatterplot.png")
plt.show()
plt.close()

## -------- example 2 
fig, ax1 = plt.subplots(1,1, figsize=[5,3.5]) # default figsize is 6x4 inches
sns.barplot(data=df, x="label", y="mae", hue="row_type")

plt.tight_layout()
plt.savefig("mae_barplot.png")
plt.show()
plt.close()

