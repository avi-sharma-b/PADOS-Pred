#activation steps:

# BASH: conda activate URAP 
# CMD + Shift + P (select conda/URAP interpreter)
# BASH: python file_name.py

import pandas as pd

# df = pd.read_csv("Chiapas_GSVDENUE_Match.csv")

#col_name = df.columns[46]
#df = df.dropna(subset = [col_name])
#df.to_csv("Clean_Cs_GSVE_Mh.csv", index=False)

# df = pd.read_csv("Clean_Cs_GSVE_Mh.csv")

# col = df.columns[4]    # Labels 


in_path = "Clean_Cs_GSVE_Mh.csv"
out_path = "Clean_Cs_GSVE_Mh_zip_plus_admin.csv"

df = pd.read_csv(in_path)

wanted = [
    "PADO",                 # label (keep, but don't use as feature)
    "zip_code",
    "zipcode_gsv",
    "cod_postal",
    "municipality_name",
    "in_target_municipality",
]

# keep only columns that actually exist
keep = [c for c in wanted if c in df.columns]
missing = [c for c in wanted if c not in df.columns]

print("Keeping:", keep)
print("Missing (not found in CSV):", missing)

df_reduced = df[keep].copy()
df_reduced.to_csv(out_path, index=False)

print("Saved:", out_path, "with shape:", df_reduced.shape)

