"""
Compare pre-policy shipment trends to help choose control states.

Only pre-policy years are used, so control choices are based on
similarity before the policy and not on post-policy results.

Outputs:
- outputs/control_state_pretrends_FL.csv
- outputs/control_state_pretrends_WA.csv
- outputs/control_state_pretrends_FL.png
- outputs/control_state_pretrends_WA.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

shipments = pd.read_parquet("opioid_shipments_per_capita.parquet")

# Policy start years. Florida's policy began in February 2010,
# so 2010 is the first post-policy year. Washington's began in January 2012.
policy_years = {"FL": 2010, "WA": 2012}

# States that cannot be controls:
# - FL and WA are the treated states.
# - TX had its own opioid regulation in January 2007.
# - DC is not a state.
not_controls = ["FL", "WA", "TX", "DC"]

# Number of top candidates to show in each plot.
n_plot = 5

os.makedirs("outputs", exist_ok=True)

# State-year MME per resident, weighted by county population.
state_year = shipments.groupby(["BUYER_STATE", "year"], as_index=False).agg(
    total_mme=("total_mme", "sum"),
    population=("population", "sum"),
)
state_year["mme_per_capita"] = state_year["total_mme"] / state_year["population"]

candidates = sorted(set(shipments["BUYER_STATE"]) - set(not_controls))

print("Candidate control states:", len(candidates))


def pretrend_test(treated, control, last_pre_year):
    """
    Test whether one control state's pre-policy trend matches the treated state.

    Uses county-year data from pre-policy years:
    mme_per_capita ~ county fixed effects + year + year * treated

    The year:treated coefficient is the difference in yearly slope
    (treated minus control). Standard errors are clustered by county.
    """
    data = shipments[
        shipments["BUYER_STATE"].isin([treated, control])
        & (shipments["year"] <= last_pre_year)
    ].copy()
    data["treated"] = (data["BUYER_STATE"] == treated).astype(int)

    model = smf.ols(
        "mme_per_capita ~ C(county_fips) + year + year:treated",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["county_fips"]})

    return model.params["year:treated"], model.pvalues["year:treated"]


for treated, policy_year in policy_years.items():
    last_pre_year = policy_year - 1
    pre = state_year[state_year["year"] <= last_pre_year]

    treated_series = pre[pre["BUYER_STATE"] == treated].set_index("year")[
        "mme_per_capita"
    ]

    rows = []
    for control in candidates:
        control_series = pre[pre["BUYER_STATE"] == control].set_index("year")[
            "mme_per_capita"
        ]

        # Remove the level difference, then measure how far apart the
        # two pre-policy paths are (root mean squared gap).
        gap = treated_series - control_series
        demeaned_gap = gap - gap.mean()
        rmse_demeaned = (demeaned_gap**2).mean() ** 0.5

        slope_diff, p_value = pretrend_test(treated, control, last_pre_year)

        rows.append(
            {
                "control_state": control,
                "pre_years": f"2006-{last_pre_year}",
                "slope_diff_mme_per_year": slope_diff,
                "slope_diff_p_value": p_value,
                "rmse_demeaned_gap": rmse_demeaned,
                "level_gap_mean": gap.mean(),
                "control_mme_per_capita_first_year": control_series.iloc[0],
                "n_counties": shipments.loc[
                    shipments["BUYER_STATE"] == control, "county_fips"
                ].nunique(),
            }
        )

    results = pd.DataFrame(rows)
    results["abs_slope_diff"] = results["slope_diff_mme_per_year"].abs()
    results = results.sort_values("abs_slope_diff").reset_index(drop=True)

    results.to_csv(f"outputs/control_state_pretrends_{treated}.csv", index=False)

    print(f"\n{treated}: pre-policy years 2006-{last_pre_year}")
    print(f"{treated} MME per resident by year:")
    print(treated_series.round(1).to_string())
    print("\nClosest pre-policy trends (smallest slope difference first):")
    print(
        results.drop(columns="abs_slope_diff").head(12).round(3).to_string(index=False)
    )

    # Plot the treated state and the closest candidates, pre-policy only.
    top = results["control_state"].head(n_plot).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        treated_series.index,
        treated_series.values,
        color="black",
        linewidth=3,
        marker="o",
        label=f"{treated} (treated)",
    )
    for control in top:
        series = pre[pre["BUYER_STATE"] == control]
        ax.plot(
            series["year"],
            series["mme_per_capita"],
            marker="o",
            label=control,
        )

    ax.set_title(f"Pre-policy MME per resident: {treated} vs closest-trend candidates")
    ax.set_xlabel("Year")
    ax.set_ylabel("MME per resident (population-weighted)")
    ax.set_xticks(treated_series.index)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(f"outputs/control_state_pretrends_{treated}.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved outputs/control_state_pretrends_{treated}.csv and .png")
