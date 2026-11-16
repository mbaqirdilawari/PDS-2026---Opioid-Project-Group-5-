import os

import matplotlib.pyplot as plt
import pandas as pd

# Plot pre-policy shipment trends for FL, WA, and possible control states.
# We only use years before each policy, so we choose controls
# based on similar trends, not on post-policy results.

shipments = pd.read_parquet("opioid_shipments_per_capita.parquet")

# State-level MME per resident for each year
state_year = shipments.groupby(["BUYER_STATE", "year"], as_index=False)[
    ["total_mme", "population"]
].sum()
state_year["mme_per_capita"] = state_year["total_mme"] / state_year["population"]

# Treated state: (last pre-policy year, candidate control states)
# Candidates have similar pre-policy trends and no major opioid policy
# change during the treated state's post-policy period.
cases = {
    "FL": (2009, ["PA", "MD", "RI", "MA"]),
    "WA": (2011, ["MN", "NE", "ND"]),
}

os.makedirs("outputs", exist_ok=True)

for treated, (last_pre_year, controls) in cases.items():
    pre = state_year[
        state_year["BUYER_STATE"].isin([treated] + controls)
        & (state_year["year"] <= last_pre_year)
    ]

    # Print the yearly values so we can compare them
    table = pre.pivot(index="year", columns="BUYER_STATE", values="mme_per_capita")
    print(f"\n{treated} and candidate controls, pre-policy MME per resident:")
    print(table.round(1))

    # Plot the treated state in black and the candidates in color
    fig, ax = plt.subplots(figsize=(8, 5))
    for state in [treated] + controls:
        data = pre[pre["BUYER_STATE"] == state]
        if state == treated:
            ax.plot(
                data["year"],
                data["mme_per_capita"],
                color="black",
                linewidth=3,
                marker="o",
                label=f"{state} (treated)",
            )
        else:
            ax.plot(data["year"], data["mme_per_capita"], marker="o", label=state)

    ax.set_title(f"Pre-policy opioid shipments: {treated} vs candidate controls")
    ax.set_xlabel("Year")
    ax.set_ylabel("MME per resident")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"outputs/control_state_pretrends_{treated}.png", dpi=150)
    plt.close(fig)
