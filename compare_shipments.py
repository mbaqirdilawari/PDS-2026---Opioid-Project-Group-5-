import pandas as pd

main = pd.read_parquet("opioid_shipments_per_capita.parquet")
sensitivity = pd.read_parquet("opioid_shipments_per_capita_with_separate.parquet")

comparison = main.merge(
    sensitivity,
    on=["BUYER_STATE", "BUYER_COUNTY", "year"],
    suffixes=("_main", "_sensitivity"),
    validate="1:1",
)

print(comparison.shape)

print((comparison["total_mme_sensitivity"] >= comparison["total_mme_main"]).all())

print(
    comparison[
        [
            "total_mme_main",
            "total_mme_sensitivity",
            "mme_per_capita_main",
            "mme_per_capita_sensitivity",
        ]
    ].head()
)
