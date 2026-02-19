import pandas as pd



# for _, row in df.iterrows():

    # print(row["merged latitude"], row["result_lng"])

#df["merged latitude"] = df["16.2381178"].combine_first(df["result_lat"])
#df.to_csv("V2_Clean_Cs_GSVE_Mh.csv", index=False)


# # latitude in "merged latitude"; longitude in "result_lng" # #



"""
from INEGI, need to extract:

Total population
Population density
Households size

Population with health insurance (derechohabiencia)
Population without health insurance

Population with disability or limitations

Literacy
Educational attainment 


Access to basic services
Housing characteristics

"""


def clean_and_zfill(s: pd.Series, width: int, *, keep_alnum: bool = True) -> pd.Series:
    
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    if not keep_alnum:
        s = s.str.replace(r"\D+", "", regex=True)
    s = s.replace("", pd.NA)
    return s.where(s.isna(), s.str.zfill(width))



# df = pd.read_csv("V2_Clean_Cs_GSVE_Mh.csv")

# df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

# df["Cve_ent"] = clean_and_zfill(df["cve_ent"], 2, keep_alnum=False)
# df["Cve_mun"] = clean_and_zfill(df["cve_mun"], 3, keep_alnum=False)
# df["Cve_loc"] = clean_and_zfill(df["cve_loc"], 4, keep_alnum=False)

# df["Ageb"] = clean_and_zfill(df["ageb"], 4, keep_alnum=True)

# df["Manzana"] = clean_and_zfill(df["manzana"], 3, keep_alnum=False)

# df["Cvegeo_ageb"] = df["Cve_ent"] + df["Cve_mun"] + df["Cve_loc"] + df["Ageb"]
# df["Cvegeo_mza"]  = df["Cvegeo_ageb"] + df["Manzana"]

# df.to_csv("V3_Clean_Cs_GSVE_Mh.csv", index=False)




# IN_PATH  = "ageb_mza_urbana_07_cpv2020/conjunto_de_datos/conjunto_de_datos_ageb_urbana_07_cpv2020.csv"  
# OUT_PATH = "chiapas_ageb_only.csv"

# # df = pd.read_csv(IN_PATH, dtype=str)


# df.columns = df.columns.str.strip()
# df["MZA"] = df["MZA"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

# df_ageb = df[df["MZA"].isin(["0", "000"])].copy()

# df_ageb.to_csv(OUT_PATH, index=False)
# print(f"Saved {len(df_ageb):,} rows to {OUT_PATH}")








############################################ V3 to V4 ############################################


# PHARM_CSV = "V3_Clean_Cs_GSVE_Mh.csv"        
# AGEB_CSV  = "chiapas_INEGI_AGEBonly.csv" 
# OUT_CSV   = "V4_Clean_Cs_GSVE_Mh.csv"


# FEATURES = [
#     "POBTOT",     
#     "PDER_SS",    
#     "PSINDER",    
#     "PCON_DISC",
#     "PCON_LIMI",
#     "P15YM_AN",
#     "GRAPROES",
#     "VPH_C_ELEC",
#     "VPH_AGUADV",
#     "VPH_DRENAJ",
#     "VPH_PISODT",
#     "PRO_OCUP_C",
# ]

# def clean(s: pd.Series) -> pd.Series:
#     return (s.astype("string")
#               .str.strip()
#               .str.replace(r"\.0$", "", regex=True))

# def zpad(s: pd.Series, n: int) -> pd.Series:
#     s = clean(s)
#     return s.where(s.isna(), s.str.zfill(n))

# ph = pd.read_csv(PHARM_CSV, dtype=str)
# ph.columns = ph.columns.str.strip()

# # pad + normalize (handles leading 0s safely)
# ph["Cve_ent"] = zpad(ph["Cve_ent"], 2)
# ph["Cve_mun"] = zpad(ph["Cve_mun"], 3)
# ph["Cve_loc"] = zpad(ph["Cve_loc"], 4)

# # AGEB can contain letters like 012A; keep uppercase and pad to 4
# ph["Ageb"] = clean(ph["Ageb"]).str.upper().str.zfill(4)

# # one-column join key (unique)
# ph["key_ageb"] = ph["Cve_ent"] + ph["Cve_mun"] + ph["Cve_loc"] + ph["Ageb"]

# inegi = pd.read_csv(AGEB_CSV, dtype=str)
# inegi.columns = inegi.columns.str.strip()

# inegi["ENTIDAD"] = zpad(inegi["ENTIDAD"], 2)
# inegi["MUN"]     = zpad(inegi["MUN"], 3)
# inegi["LOC"]     = zpad(inegi["LOC"], 4)
# inegi["AGEB"]    = clean(inegi["AGEB"]).str.upper().str.zfill(4)
# inegi["MZA"]     = zpad(inegi["MZA"], 3)

# ageb_totals = inegi[(inegi["MZA"] == "000") & (inegi["AGEB"] != "0000")].copy()

# ageb_totals["key_ageb"] = (
#     ageb_totals["ENTIDAD"] + ageb_totals["MUN"] + ageb_totals["LOC"] + ageb_totals["AGEB"]
# )

# # keep only requested features that exist
# FEATURES = [c for c in FEATURES if c in ageb_totals.columns]
# if "POBTOT" not in FEATURES:
#     print("WARNING: POBTOT not found in INEGI columns. Check your file/headers.")

# ageb_small = ageb_totals[["key_ageb"] + FEATURES].copy()

# # numeric conversion (optional but usually desired for ML)
# for c in FEATURES:
#     ageb_small[c] = pd.to_numeric(ageb_small[c], errors="coerce")

# # sanity: key should be unique on INEGI side
# dups = ageb_small["key_ageb"].duplicated().sum()
# if dups:
#     # show a few duplicates to debug
#     print("ERROR: INEGI key_ageb not unique. Example duplicate keys:")
#     print(ageb_small.loc[ageb_small["key_ageb"].duplicated(keep=False), "key_ageb"].head(20).to_list())
#     raise ValueError("INEGI key_ageb not unique even after filtering (MZA==000 and AGEB!=0000).")


# out = ph.merge(ageb_small, on="key_ageb", how="left")

# #chekcer
# print("Rows in original:", len(ph))
# print("Rows after merge:", len(out))
# if "POBTOT" in out.columns:
#     print("Match rate (POBTOT non-null):", out["POBTOT"].notna().mean())

# out.to_csv(OUT_CSV, index=False)
# print("Saved:", OUT_CSV)





############################################ V4 to V5 ############################################


ENRICHED_CSV = "V4_Clean_Cs_GSVE_Mh.csv"        
AGEB_CSV  = "chiapas_INEGI_AGEBonly.csv" 
OUT_CSV   = "V5_Clean_Cs_GSVE_Mh.csv"

def clean(s: pd.Series) -> pd.Series:
    return (s.astype("string")
              .str.strip()
              .str.replace(r"\.0$", "", regex=True))

def zpad(s: pd.Series, n: int) -> pd.Series:
    s = clean(s)
    return s.where(s.isna(), s.str.zfill(n))


ph = pd.read_csv(ENRICHED_CSV, dtype=str)
ph.columns = ph.columns.str.strip()

# rebuild key_ageb (same as before)
ph["Cve_ent"] = zpad(ph["Cve_ent"], 2)
ph["Cve_mun"] = zpad(ph["Cve_mun"], 3)
ph["Cve_loc"] = zpad(ph["Cve_loc"], 4)
ph["Ageb"]    = clean(ph["Ageb"]).str.upper().str.zfill(4)
ph["key_ageb"] = ph["Cve_ent"] + ph["Cve_mun"] + ph["Cve_loc"] + ph["Ageb"]


inegi = pd.read_csv(AGEB_CSV, dtype=str)
inegi.columns = inegi.columns.str.strip()

inegi["ENTIDAD"] = zpad(inegi["ENTIDAD"], 2)
inegi["MUN"]     = zpad(inegi["MUN"], 3)
inegi["LOC"]     = zpad(inegi["LOC"], 4)
inegi["AGEB"]    = clean(inegi["AGEB"]).str.upper().str.zfill(4)
inegi["MZA"]     = zpad(inegi["MZA"], 3)

ageb_totals = inegi[(inegi["MZA"] == "000") & (inegi["AGEB"] != "0000")].copy()
ageb_totals["key_ageb"] = ageb_totals["ENTIDAD"] + ageb_totals["MUN"] + ageb_totals["LOC"] + ageb_totals["AGEB"]

if "TOTHOG" not in ageb_totals.columns:
    raise ValueError("TOTHOG not found in INEGI file columns. Check you downloaded the AGEB+MZA CPV2020 product.")

ageb_households = ageb_totals[["key_ageb", "TOTHOG"]].copy()
ageb_households["TOTHOG"] = pd.to_numeric(ageb_households["TOTHOG"], errors="coerce")


if ageb_households["key_ageb"].duplicated().any():
    raise ValueError("INEGI key_ageb not unique after filtering (MZA==000 & AGEB!=0000).")


out = ph.merge(ageb_households, on="key_ageb", how="left")

out.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print("Missing rate for TOTHOG:", out["TOTHOG"].isna().mean())





"""
POBTOT = Total population
PDER_SS = Population affiliated to health services
PSINDER = Population without affiliation to health services


PCON_DISC = Population with disability
PCON_LIMI = Population with limitation

TOTHOG = Total count of households

VPH_C_ELEC = Occupied private dwellings with electricity
VPH_AGUADV = Occupied private dwellings with piped water available
VPH_DRENAJ = Occupied private dwellings with drainage/sewer
VPH_PISODT = Occupied private dwellings with floor material other than dirt

PRO_OCUP_C = Average occupants per room
GRAPROES = Average years/grade of schooling (15+)

******rates********:


Initial features below 

Insured Rate: PDER_SS/POBTOT
Disability Rate: (PCON_DISC + PCON_LIMI)/POBTOT
Electricity Access Rate: VPH_C_ELEC/POBTOT
Piped Water Access Rate: VPH_AGUADV/POBTOT
Drainage Access Rate: VPH_DRENAJ/POBTOT
Floor Material Rate: VPH_PISODT/POBTOT
Occupants per Room: PRO_OCUP_C
Average Schooling: GRAPROES


"""

