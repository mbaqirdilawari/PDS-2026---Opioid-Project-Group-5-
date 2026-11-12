import pandas as pd
import math

shipments = pd.read_parquet("arcos_main.parquet")
separate = pd.read_parquet("arcos_separate.parquet")

print("Main data shape:")
print(shipments.shape)

print("\nDrugs:")
print(shipments["DRUG_NAME"].value_counts())

print("\nBuyer types in main data:")
print(shipments["BUYER_BUS_ACT"].value_counts())

print("\nBuyer types kept separately:")
print(separate["BUYER_BUS_ACT"].value_counts())

# Keep US states and DC only.
# Alaska is excluded as required by the project instructions.
# Territories are excluded because the analysis compares US states
# and requires consistent county population and mortality data.
valid_states = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

print("\nExtra geographic codes:")
print(
    shipments.loc[
        ~shipments["BUYER_STATE"].isin(valid_states),
        ["BUYER_STATE", "BUYER_COUNTY"],
    ].drop_duplicates()
)

shipments = shipments[shipments["BUYER_STATE"].isin(valid_states)].copy()

# Sum across opioid drugs and buyer types.
county_year_shipments = shipments.groupby(
    ["BUYER_STATE", "BUYER_COUNTY", "year"],
    as_index=False,
).agg(
    total_mme=("MME", "sum"),
    missing_mme_records=("MISSING_MME", "sum"),
)

assert math.isclose(
    county_year_shipments["total_mme"].sum(),
    shipments["MME"].sum(),
)

print("\nCounty-year shipment data:")
print(county_year_shipments.head())

print("\nShape:")
print(county_year_shipments.shape)

print("\nDuplicate county-year rows:")
print(
    county_year_shipments.duplicated(
        subset=["BUYER_STATE", "BUYER_COUNTY", "year"]
    ).sum()
)

print("\nNegative MME totals:")
print((county_year_shipments["total_mme"] < 0).sum())

print("\nState codes:")
print(sorted(county_year_shipments["BUYER_STATE"].unique()))

print("\nNumber of state codes:")
print(county_year_shipments["BUYER_STATE"].nunique())

# Keep the same geographic sample in the separate-facility data.
separate = separate[separate["BUYER_STATE"].isin(valid_states)].copy()

# Add treatment facilities and central-fill pharmacies
# to create a sensitivity-analysis version.
shipments_with_separate = pd.concat(
    [shipments, separate],
    ignore_index=True,
)

county_year_with_separate = shipments_with_separate.groupby(
    ["BUYER_STATE", "BUYER_COUNTY", "year"],
    as_index=False,
).agg(
    total_mme=("MME", "sum"),
    missing_mme_records=("MISSING_MME", "sum"),
)

assert math.isclose(
    county_year_with_separate["total_mme"].sum(),
    shipments_with_separate["MME"].sum(),
)

print("\nSensitivity data shape:")
print(county_year_with_separate.shape)

print("\nSensitivity duplicate county-year rows:")
print(
    county_year_with_separate.duplicated(
        subset=["BUYER_STATE", "BUYER_COUNTY", "year"]
    ).sum()
)

print("\nPrimary total MME:")
print(county_year_shipments["total_mme"].sum())

print("\nSensitivity total MME:")
print(county_year_with_separate["total_mme"].sum())

# Validate the final datasets.
county_year_key = ["BUYER_STATE", "BUYER_COUNTY", "year"]

assert not county_year_shipments.duplicated(subset=county_year_key).any()

assert not county_year_with_separate.duplicated(subset=county_year_key).any()

assert (county_year_shipments["total_mme"] >= 0).all()
assert (county_year_with_separate["total_mme"] >= 0).all()

assert county_year_shipments["year"].between(2006, 2015).all()
assert county_year_with_separate["year"].between(2006, 2015).all()

assert (
    county_year_with_separate["total_mme"].sum()
    >= county_year_shipments["total_mme"].sum()
)

# Save the two county-year datasets.
county_year_shipments.to_parquet(
    "arcos_county_year.parquet",
    index=False,
)

county_year_with_separate.to_parquet(
    "arcos_county_year_with_separate.parquet",
    index=False,
)

print("\nValidation passed.")
print("Saved both county-year Parquet files.")
