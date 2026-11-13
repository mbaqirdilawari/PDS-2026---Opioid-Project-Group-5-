import pandas as pd

# Load county-year shipment and population data
shipments = pd.read_parquet("arcos_county_year_with_separate.parquet")
population = pd.read_parquet("county_population_2006_2015.parquet")

print("Shipment data:")
print(shipments.head())
print(shipments.shape)

print("\nPopulation data:")
print(population.head())
print(population.shape)

print("\nShipment state examples:")
print(sorted(shipments["BUYER_STATE"].unique())[:10])

print("\nShipment county examples:")
print(shipments["BUYER_COUNTY"].head(10).tolist())

print("\nPopulation county examples:")
print(population["CTYNAME"].head(10).tolist())

# Match full state names to abbreviations
state_abbreviations = {
    "Alabama": "AL",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

population["state_abbr"] = population["STNAME"].map(state_abbreviations)

# Clean county names in the population data
population["county_name"] = (
    population["CTYNAME"]
    .str.upper()
    .str.replace(" COUNTY", "", regex=False)
    .str.replace(" PARISH", "", regex=False)
    .str.replace(" BOROUGH", "", regex=False)
    .str.replace(" CENSUS AREA", "", regex=False)
    .str.replace(" MUNICIPALITY", "", regex=False)
    .str.replace(r"^ST\. ", "SAINT ", regex=True)
    .str.replace(r"^STE\. ", "SAINTE ", regex=True)
    .str.replace("'", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace("Ñ", "N", regex=False)
    .str.replace(" ", "", regex=False)
)

# Match two Virginia independent cities in the shipment data
population.loc[
    (population["state_abbr"] == "VA")
    & population["county_name"].isin(["RADFORDCITY", "SALEMCITY"]),
    "county_name",
] = population.loc[
    (population["state_abbr"] == "VA")
    & population["county_name"].isin(["RADFORDCITY", "SALEMCITY"]),
    "county_name",
].str.replace(
    "CITY", "", regex=False
)

# Clean county names in the shipment data
shipments["shipment_county_name"] = (
    shipments["BUYER_COUNTY"]
    .str.upper()
    .str.replace(r"^ST ", "SAINT ", regex=True)
    .str.replace(r"^STE ", "SAINTE ", regex=True)
    .str.replace("'", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace("Ñ", "N", regex=False)
    .str.replace(" ", "", regex=False)
)


print("\nPopulation names after cleaning:")
print(population[["STNAME", "state_abbr", "CTYNAME", "county_name"]].head())

print("\nMissing state abbreviations:")
print(population.loc[population["state_abbr"].isna(), "STNAME"].unique())

# Check whether shipment and population data match
merged = pd.merge(
    shipments,
    population[["county_fips", "state_abbr", "county_name", "year", "population"]],
    left_on=["BUYER_STATE", "shipment_county_name", "year"],
    right_on=["state_abbr", "county_name", "year"],
    how="outer",
    validate="1:1",
    indicator=True,
)

print("\nMerge results:")
print(merged["_merge"].value_counts())

print("\nUnmatched shipment counties:")
print(
    merged.loc[
        merged["_merge"] == "left_only",
        ["BUYER_STATE", "BUYER_COUNTY"],
    ]
    .drop_duplicates()
    .sort_values(["BUYER_STATE", "BUYER_COUNTY"])
)

print("\nUnmatched population counties:")
print(
    merged.loc[
        merged["_merge"] == "right_only",
        ["state_abbr", "county_name"],
    ]
    .drop_duplicates()
    .sort_values(["state_abbr", "county_name"])
)

print(merged["_merge"].value_counts())

# Check unmatched shipment counties
unmatched_shipments = (
    merged.loc[
        merged["_merge"] == "left_only",
        ["BUYER_STATE", "BUYER_COUNTY"],
    ]
    .drop_duplicates()
    .sort_values(["BUYER_STATE", "BUYER_COUNTY"])
)

print("\nUnmatched shipment counties:")
print(unmatched_shipments.to_string(index=False))
print("\nNumber of unmatched county names:")
print(len(unmatched_shipments))

# Keep successfully matched county-year observations
shipment_population = merged.loc[merged["_merge"] == "both"].copy()

# Calculate MME per resident
shipment_population["mme_per_capita"] = (
    shipment_population["total_mme"] / shipment_population["population"]
)

# Keep the useful columns
shipment_population = shipment_population[
    [
        "BUYER_STATE",
        "BUYER_COUNTY",
        "county_fips",
        "year",
        "total_mme",
        "population",
        "mme_per_capita",
        "missing_mme_records",
    ]
]

print("\nFinal shipment-population data:")
print(shipment_population.head())
print(shipment_population.shape)

# Validate final data
assert not shipment_population.duplicated(["BUYER_STATE", "BUYER_COUNTY", "year"]).any()

assert shipment_population["population"].notna().all()
assert (shipment_population["population"] > 0).all()
assert (shipment_population["mme_per_capita"] >= 0).all()

# Save final data
shipment_population.to_parquet(
    "opioid_shipments_per_capita_with_separate.parquet",
    index=False,
)

print("\nSaved opioid_shipments_per_capita_with_separate.parquet")
