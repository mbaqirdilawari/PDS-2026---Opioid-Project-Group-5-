import pandas as pd
import pyarrow  # Required for saving Parquet files.

columns = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "BUYER_BUS_ACT",
    "TRANSACTION_CODE",
    "TRANSACTION_DATE",
    "DRUG_NAME",
    "MME"
]

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

totals = None
missing_locations = 0
missing_mme = 0

# Read the entire file, 100,000 rows at a time.
chunks = pd.read_csv(
    "arcos_all.tsv",
    sep="\t",
    usecols=columns,
    chunksize=100_000
)

for number, arcos in enumerate(chunks, start=1):

    # Keep sales rows only. Indicated by Transaction code "S"
    arcos = arcos[
        (arcos["TRANSACTION_CODE"] == "S")
        & (arcos["BUYER_BUS_ACT"].isin(keep_buyers + separate_buyers))
    ].copy()

    # Extract the year from each transaction date.
    arcos["year"] = pd.to_datetime(arcos["TRANSACTION_DATE"]).dt.year

    # Keep the shared shipment/mortality period and exclude Alaska.
    arcos = arcos[
        (arcos["year"] >= 2006)
        & (arcos["year"] <= 2015)
        & (arcos["BUYER_STATE"] != "AK")
    ].copy()

    # find missing values
    missing_locations += (
        arcos["BUYER_STATE"].isna()
        | arcos["BUYER_COUNTY"].isna()
    ).sum()

    # remove records that dont have a county assignment
    arcos = arcos.dropna(
        subset=["BUYER_STATE", "BUYER_COUNTY"]
    ).copy()

    # Convert shipment amounts to numbers.
    # Missing or invalid amounts are flagged rather than stopping the run.
    numeric_mme = pd.to_numeric(arcos["MME"], errors="coerce")

    arcos["MISSING_MME"] = numeric_mme.isna().astype(int)
    missing_mme += arcos["MISSING_MME"].sum()

    # Save affected records with their original MME values for inspection.
    if arcos["MISSING_MME"].sum() > 0:
        arcos[arcos["MISSING_MME"] == 1].to_csv(
            f"arcos_missing_mme_chunk_{number}.csv",
            index=False
        )

    arcos["MME"] = numeric_mme

    #Total up shipments for buyer and drug categories
    chunk_totals = arcos.groupby(
        ["BUYER_STATE", "BUYER_COUNTY", "year", "BUYER_BUS_ACT", "DRUG_NAME"],
        as_index=False,
        dropna=False
    )[["MME", "MISSING_MME"]].sum(min_count=1)

    # Combine this chunk's summary with the running totals.
    if totals is None:
        totals = chunk_totals
    else:
        totals = pd.concat(
            [totals, chunk_totals],
            ignore_index=True
        )

        totals = totals.groupby(
            ["BUYER_STATE", "BUYER_COUNTY", "year", "BUYER_BUS_ACT", "DRUG_NAME"],
            as_index=False,
            dropna=False
        )[["MME", "MISSING_MME"]].sum(min_count=1)

    # Preserve partial results every 100 chunks.
    if number % 100 == 0:
        totals.to_parquet("arcos_checkpoint.parquet", index=False)

        pd.DataFrame({
            "last_completed_chunk": [number],
            "missing_locations": [missing_locations],
            "missing_mme": [missing_mme]
        }).to_csv("arcos_checkpoint_info.csv", index=False)

    print(f"Finished chunk {number}", flush=True)

# Separate the main analysis recipient facilities from the separated facilities
main_totals = totals[
    totals["BUYER_BUS_ACT"].isin(keep_buyers)
].copy()

separate_totals = totals[
    totals["BUYER_BUS_ACT"].isin(separate_buyers)
].copy()

main_totals.to_parquet("arcos_main.parquet", index=False)
separate_totals.to_parquet("arcos_separate.parquet", index=False)

# Save the final processing counts.
pd.DataFrame({
    "last_completed_chunk": [number],
    "missing_locations": [missing_locations],
    "missing_mme": [missing_mme]
}).to_csv("arcos_processing_summary.csv", index=False)

print("\nRows excluded for missing location:", missing_locations)
print("Rows with missing or invalid MME:", missing_mme)
print("Saved arcos_main.parquet and arcos_separate.parquet")