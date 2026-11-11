import pandas as pd

# Read the first 100,000 rows to inspect the data before processing
# the full ARCOS file.
columns = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "BUYER_BUS_ACT",
    "TRANSACTION_CODE",
    "TRANSACTION_DATE",
    "DRUG_NAME",
    "MME"
]

sample = pd.read_csv(
    "arcos_all.tsv",
    sep="\t",
    usecols=columns,
    nrows=100_000
)

# Inspect recipient types, transaction types, and drugs.
print("\nBUYER TYPES:")
print(sample["BUYER_BUS_ACT"].value_counts())

print("\nTRANSACTION TYPES:")
print(sample["TRANSACTION_CODE"].value_counts())

print("\nDRUGS:")
print(sample["DRUG_NAME"].value_counts())

# Check whether information needed for county/year totals is missing.
print("\nMISSING VALUES:")
print(sample.isna().sum())

print("\nROWS WITH MISSING COUNTY:")
print(sample[sample["BUYER_COUNTY"].isna()])

# Extract the year for annual shipment totals.
sample["year"] = pd.to_datetime(sample["TRANSACTION_DATE"]).dt.year

print("\nYEARS IN SAMPLE:")
print(sample["year"].value_counts().sort_index())

# Inspect how recipient types overlap with transaction types.
print("\nBUYER TYPES BY TRANSACTION:")
print(pd.crosstab(
    sample["BUYER_BUS_ACT"],
    sample["TRANSACTION_CODE"]
).to_string())

# Test county/year aggregation.
# These are incomplete sample totals and include unfiltered transactions.
# They are not used as final analysis results.
county_year_sample = sample.groupby(
    ["BUYER_STATE", "BUYER_COUNTY", "year"],
    as_index=False
)["MME"].sum()

print("\nPRACTICE COUNTY/YEAR TOTALS:")
print(county_year_sample.head())