import pandas as pd

population_2000_2010 = pd.read_csv(
    "data/county_population_2000_2010.csv",
    encoding="latin-1",
)

population_2010_2019 = pd.read_csv(
    "data/county_population_2010_2019.csv",
    encoding="latin-1",
)

print("2000–2010 shape:")
print(population_2000_2010.shape)

print("\n2000–2010 columns:")
print(population_2000_2010.columns.tolist())

print("\n2000–2010 first rows:")
print(population_2000_2010.head())

print("\n2010–2019 shape:")
print(population_2010_2019.shape)

print("\n2010–2019 columns:")
print(population_2010_2019.columns.tolist())

print("\n2010–2019 first rows:")
print(population_2010_2019.head())


# Keep county-level rows only.
pop_early = population_2000_2010.loc[population_2000_2010["SUMLEV"] == 50].copy()

# Create 5-digit county FIPS codes.
pop_early["county_fips"] = pop_early["STATE"].astype(str).str.zfill(2) + pop_early[
    "COUNTY"
].astype(str).str.zfill(3)

# Keep only the years needed from this file.
early_columns = [
    "county_fips",
    "STNAME",
    "CTYNAME",
    "POPESTIMATE2006",
    "POPESTIMATE2007",
    "POPESTIMATE2008",
    "POPESTIMATE2009",
]

# Population data for 2006–2009
pop_early = pop_early[early_columns]

print("\nEarly population data:")
print(pop_early.head())
print(pop_early.shape)
print(pop_early["county_fips"].head())

pop_early = pop_early.melt(
    id_vars=["county_fips", "STNAME", "CTYNAME"],
    value_vars=[
        "POPESTIMATE2006",
        "POPESTIMATE2007",
        "POPESTIMATE2008",
        "POPESTIMATE2009",
    ],
    var_name="year",
    value_name="population",
)

pop_early["year"] = (
    pop_early["year"].str.replace("POPESTIMATE", "", regex=False).astype(int)
)

print("\nEarly population in long format:")
print(pop_early.head())
print(pop_early.shape)
print(sorted(pop_early["year"].unique()))

# Population data for 2010–2015
pop_late = population_2010_2019.loc[population_2010_2019["SUMLEV"] == 50].copy()

pop_late["county_fips"] = pop_late["STATE"].astype(str).str.zfill(2) + pop_late[
    "COUNTY"
].astype(str).str.zfill(3)

pop_late = pop_late.melt(
    id_vars=["county_fips", "STNAME", "CTYNAME"],
    value_vars=[
        "POPESTIMATE2010",
        "POPESTIMATE2011",
        "POPESTIMATE2012",
        "POPESTIMATE2013",
        "POPESTIMATE2014",
        "POPESTIMATE2015",
    ],
    var_name="year",
    value_name="population",
)

pop_late["year"] = (
    pop_late["year"].str.replace("POPESTIMATE", "", regex=False).astype(int)
)

print("\nLate population in long format:")
print(pop_late.head())
print(pop_late.shape)
print(sorted(pop_late["year"].unique()))

# Combine population data for 2006–2015
population = pd.concat(
    [pop_early, pop_late],
    ignore_index=True,
)

# Exclude Alaska to match the shipment data
population = population[population["STNAME"] != "Alaska"].copy()

print("\nCombined population data:")
print(population.head())
print(population.shape)
print(sorted(population["year"].unique()))
print(population.duplicated(["county_fips", "year"]).sum())

# Check the final population data
assert population["year"].between(2006, 2015).all()
assert not population.duplicated(["county_fips", "year"]).any()
assert (population["population"] > 0).all()
assert "Alaska" not in population["STNAME"].unique()

# Save the cleaned population data
population.to_parquet(
    "county_population_2006_2015.parquet",
    index=False,
)

print("\nSaved county_population_2006_2015.parquet")
