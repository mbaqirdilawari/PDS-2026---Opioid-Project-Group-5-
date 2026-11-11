import pandas as pd

for filename in ["arcos_main.parquet", "arcos_separate.parquet"]:

    shipments = pd.read_parquet(filename)

    print("\nChecking:", filename)

    print("\nStates:")
    print(sorted(shipments["BUYER_STATE"].unique()))
    print(f"\nTotal number of states in the data is {len(shipments["BUYER_STATE"].unique())}")

    print("\nDrugs:")
    print(sorted(shipments["DRUG_NAME"].unique()))

    print("\nYears:")
    print(sorted(shipments["year"].unique()))

    print("\nMissing values:")
    print(shipments.isna().sum())

    print("\nRows with negative MME:")
    print((shipments["MME"] < 0).sum())

    print("\nDuplicate summary rows:")
    print(shipments.duplicated(
        subset=[
            "BUYER_STATE",
            "BUYER_COUNTY",
            "year",
            "BUYER_BUS_ACT",
            "DRUG_NAME"
        ]
    ).sum())

    print("\nTransactions with missing or invalid MME:")
    print(shipments["MISSING_MME"].sum())