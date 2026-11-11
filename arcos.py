import pandas as pd

columns = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "BUYER_BUS_ACT",
    "TRANSACTION_CODE",
    "TRANSACTION_DATE",
    "DRUG_NAME",
    "MME"
]

# Start with small data portion to devlop the code cleaning process
arcos = pd.read_csv(
    "arcos_all.tsv",
    sep = "\t",
    usecols = columns,
    nrows = 200_000

)

""" List of buyer types to keep. Excludes facilities like manufacturers, distributors, importers,
exporters, labs buying drugs for testing, and researchers using drugs for research. Common theme is to exclude 
entities that are not buying to distribute to patients. Addiction recovery centers and central fill pharmacies
are separated in order to evaluate their affect on the results separately in the final analysis.
"""

keep_buyers = [
    "CHAIN PHARMACY",
    "RETAIL PHARMACY",
    "HOSPITAL/CLINIC",
    "HOSP/CLINIC-VA",
    "PRACTITIONER",
    "MLP-NURSE PRACTITIONER",
    "PRACTITIONER-DW/30",
    "PRACTITIONER-DW/100",
    "PRACTITIONER-DW/275"
]

separate_buyers = [
    "MAINT & DETOX",
    "COMP/MAINT/DETOX",
    "CENTRAL FILL PHARMACY"
]

# Keep sales rows only. Indicated by Transaction code "S"
arcos = arcos[arcos["TRANSACTION_CODE"] == "S"].copy()

# Separate the main analysis recipient facilities from the separated facilities

main_shipments = arcos[
    arcos["BUYER_BUS_ACT"].isin(keep_buyers)
].copy()

separate_shipments = arcos[
    arcos["BUYER_BUS_ACT"].isin(separate_buyers)
].copy()

print("Main buyer types:")
print(main_shipments["BUYER_BUS_ACT"].value_counts())

print("\nSeparated buyer types:")
print(separate_shipments["BUYER_BUS_ACT"].value_counts())


# clean both datasets the same way

for data in [main_shipments, separate_shipments]:

    # Extract the year from each transaction date.
    data["year"] = pd.to_datetime(data["TRANSACTION_DATE"]).dt.year

    # find missing values
    print("\nMissing locations values:")
    print(data[["BUYER_STATE", "BUYER_COUNTY"]].isna().sum())

# remove records that dont have a county assignment
main_shipments = main_shipments.dropna(
    subset = ["BUYER_STATE", "BUYER_COUNTY"]
)

separate_shipments = separate_shipments.dropna(
    subset = ["BUYER_STATE", "BUYER_COUNTY"]
)

# Keep the shared shipment/mortality period and exclude Alaska.
main_shipments = main_shipments[
    (main_shipments["year"] >= 2006)
    & (main_shipments["year"] <= 2015)
    & (main_shipments["BUYER_STATE"] != "AK")
].copy()

separate_shipments = separate_shipments[
    (separate_shipments["year"] >= 2006)
    & (separate_shipments["year"] <= 2015)
    & (separate_shipments["BUYER_STATE"] != "AK")
].copy()

#Total up shipments for buyer and drug categories
main_totals = main_shipments.groupby(
    ["BUYER_STATE", "BUYER_COUNTY", "year", "BUYER_BUS_ACT", "DRUG_NAME"],
    as_index = False,
    dropna =False
                                    )["MME"].sum(min_count = 1)

separate_totals = separate_shipments.groupby(
    ["BUYER_STATE", "BUYER_COUNTY", "year", "BUYER_BUS_ACT", "DRUG_NAME"],
    as_index = False,
    dropna =False
                                    )["MME"].sum(min_count = 1)

print("\nMain shipment totals — EXAMPLE DATA:")
print(main_totals.head())

print("\nSeparate shipment totals — EXAMPLE DATA:")
print(separate_totals.head())