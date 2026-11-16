# Source: CDC WONDER, Underlying Cause of Death, 1999–2020.
# https://wonder.cdc.gov/ucd-icd10.html
# Separate download accessed November 15, 2026; not the annual files
# supplied with the assignment.
#
# Query: County and Year, 2006–2015, all demographic groups,
# combining overdose categories D1–D4. Zero and suppressed values included.
# These are all-drug overdose deaths, not opioid-specific deaths.
#
# CDC reporting caveats:
# Georgia, 2008–2009: unusually high ill-defined-cause (R99) counts.
# New Jersey, 2009: unusually high R99 counts and underreported counts
# for other causes, particularly unintentional poisonings.
# Consider these issues when selecting controls and interpreting trends.

import pandas as pd

# Read combined overdose deaths: D1, D2, D3, and D4 together.
# Read codes as text to preserve leading zeros in county FIPS.
mortality = pd.read_csv("data/overdose_county_year_2006_2015.tsv", sep="\t", dtype=str)

# Convert years to numbers. Footer notes become missing values.
mortality["year"] = pd.to_numeric(mortality["Year"], errors="coerce")

# Keep the analysis years and remove footer notes.
mortality = mortality[
    mortality["year"].between(2006, 2015) & mortality["County Code"].notna()
].copy()

mortality["year"] = mortality["year"].astype(int)

# Format county FIPS to match the shipment and population data.
mortality["county_fips"] = mortality["County Code"].str.zfill(5)

# Exclude Alaska, whose state FIPS code is 02.
mortality = mortality[~mortality["county_fips"].str.startswith("02")].copy()

# Preserve the original death values before converting them.
# This distinguishes suppressed counts from other missing values.
mortality["deaths_original"] = mortality["Deaths"].str.strip()

mortality["count_suppressed"] = mortality["deaths_original"] == "Suppressed"

# Keep reported numbers, including actual zeros.
# Text such as "Suppressed" or "Missing" becomes NaN.
mortality["overdose_deaths"] = pd.to_numeric(
    mortality["deaths_original"], errors="coerce"
)

# Flag unavailable counts separately from suppressed counts.
mortality["count_unavailable"] = (
    mortality["overdose_deaths"].isna() & ~mortality["count_suppressed"]
)

# Keep the columns needed for checking and later analysis.
mortality = mortality[
    [
        "county_fips",
        "County",
        "year",
        "overdose_deaths",
        "deaths_original",
        "count_suppressed",
        "count_unavailable",
    ]
].copy()

# Check that each county/year appears only once.
duplicate_rows = mortality.duplicated(subset=["county_fips", "year"]).sum()

print("County/year rows:", len(mortality))
print("Duplicate county/year rows:", duplicate_rows)

print("\nCount status:")
print("Reported positive counts:", (mortality["overdose_deaths"] > 0).sum())
print("Confirmed zeros:", (mortality["overdose_deaths"] == 0).sum())
print("Suppressed counts:", mortality["count_suppressed"].sum())
print("Other unavailable counts:", mortality["count_unavailable"].sum())

print("\nCounty/year rows by year:")
print(mortality["year"].value_counts().sort_index())

print("\nExamples:")
print(mortality.head(10).to_string(index=False))

# Prevent saving a table with multiple records for the same county/year.
if duplicate_rows > 0:
    raise ValueError(
        "Duplicate county/year rows found. Check that the CDC download "
        "was grouped by County and Year only."
    )


# Read county population for the analysis years.
population = pd.read_parquet("county_population_2006_2015.parquet")

# Match the county FIPS format used in the mortality table.
population["county_fips"] = population["county_fips"].astype(str).str.zfill(5)

# Match Oglala Lakota's population code to the Shannon code
# used in the mortality data. This is the same county.
population.loc[population["county_fips"] == "46102", "county_fips"] = "46113"

# Bedford's population source combines city and county geography
# from 2010 onward, while mortality still separates them.
# Exclude these mismatched county/year records from this analysis.
population = population[
    ~(population["county_fips"].isin(["51019", "51515"]) & (population["year"] >= 2010))
].copy()

# Attach population using county FIPS and year.
mortality = mortality.merge(
    population[["county_fips", "year", "population"]],
    on=["county_fips", "year"],
    how="left",
    validate="one_to_one",
)

# Identify rows that cannot be used for estimation.
# Suppressed counts remain eligible when population is available.
cannot_use = (
    mortality["count_unavailable"]
    | mortality["population"].isna()
    | (mortality["population"] <= 0)
)

# Save excluded records so we can document exactly what was removed.
excluded = mortality[cannot_use].copy()

excluded.to_csv("mortality_excluded_records.csv", index=False)

mortality = mortality[~cannot_use].copy()

# Identify each county's state from the first two FIPS digits.
mortality["state_fips"] = mortality["county_fips"].str[:2]

print("\nRecords excluded:")
print(
    excluded[["County", "year", "deaths_original", "population"]].to_string(index=False)
)

print("\nRows available for the mortality model:", len(mortality))
print("Suppressed counts to estimate:", mortality["count_suppressed"].sum())
print("Missing population:", mortality["population"].isna().sum())

# Compare the share of suppressed rows with the share of population
# living in those county/year observations.
for state_name, state_code in [("Florida", "12"), ("Washington", "53")]:

    state_data = mortality[
        mortality["state_fips"] == state_code
    ]

    suppressed_data = state_data[
        state_data["count_suppressed"]
    ]

    row_share = (
        len(suppressed_data) / len(state_data) * 100
    )

    population_share = (
        suppressed_data["population"].sum()
        / state_data["population"].sum()
        * 100
    )

    print(f"\n{state_name} suppression:")
    print(f"County/year rows suppressed: {row_share:.1f}%")
    print(f"Population share in suppressed rows: {population_share:.1f}%")

# Save reported counts and missing-count flags without estimating deaths.
mortality.to_parquet("mortality_county_year.parquet", index=False)

print("\nSaved mortality_county_year.parquet")
