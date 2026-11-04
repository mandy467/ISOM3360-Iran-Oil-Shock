"""Merge the 8 raw CSVs into one country-year panel.

Run from the repo root:  python src/build_panel.py
Output: data/processed/panel.csv  (one row per country per year, 2010-2025)

Missing values are kept as NaN on purpose: in this dataset missing != zero.
"""
import numpy as np
import pandas as pd

from config import RAW, PROCESSED, INFLATION_STRESS_THRESHOLD


def load():
    country = pd.read_csv(RAW / "01_country_master.csv")
    energy = pd.read_csv(RAW / "02_energy_monthly_master.csv")
    market = pd.read_csv(RAW / "03_global_market_monthly_master.csv")
    macro = pd.read_csv(RAW / "04_macro_annual_master.csv")
    trade = pd.read_csv(RAW / "05_oil_trade_master.csv")
    cons = pd.read_csv(RAW / "06_oil_consumption_master.csv")
    fx = pd.read_csv(RAW / "07_fx_annual_master_historical.csv")
    return country, energy, market, macro, trade, cons, fx


def build():
    country, energy, market, macro, trade, cons, fx = load()

    # Clean country metadata (some region labels have trailing spaces)
    country = country[["iso3", "country_name", "region", "income_group"]].copy()
    country["region"] = country["region"].str.strip()

    # Monthly -> annual: production (mean barrels/day for the year)
    energy["year"] = energy["period"].str[:4].astype(int)
    prod = (energy.groupby(["iso3", "year"])[["production_total_tbpd", "production_crude_tbpd"]]
                  .mean().reset_index())

    # Monthly -> annual: global oil price (same for every country)
    market["year"] = pd.to_datetime(market["period"]).dt.year
    price = (market.groupby("year")
                   .agg(brent_mean=("brent_usd_bbl", "mean"),
                        brent_max=("brent_usd_bbl", "max"),
                        brent_std=("brent_usd_bbl", "std"),
                        wti_mean=("wti_usd_bbl", "mean"))
                   .reset_index())
    price["brent_yoy_pct"] = price["brent_mean"].pct_change() * 100

    # Skeleton: every country x every year 2010-2025
    years = range(2010, 2026)
    panel = pd.MultiIndex.from_product([country["iso3"], years], names=["iso3", "year"]).to_frame(index=False)
    panel = panel.merge(country, on="iso3", how="left")

    drop_name = lambda d: d.drop(columns=["country_name"], errors="ignore")
    for df in [macro, trade, cons, fx]:
        panel = panel.merge(drop_name(df), on=["iso3", "year"], how="left")
    panel = panel.merge(prod, on=["iso3", "year"], how="left")
    panel = panel.merge(price, on="year", how="left")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    g = panel.groupby("iso3")

    # Derived exposure features (same definitions as file 08, but for every year)
    panel["net_crude_trade_usd"] = panel["crude_oil_import_usd"] - panel["crude_oil_export_usd"]
    panel["import_burden_pct_gdp"] = panel["crude_oil_import_usd"] / panel["gdp_usd"] * 100
    panel["export_dependence_pct_gdp"] = panel["crude_oil_export_usd"] / panel["gdp_usd"] * 100
    panel["oil_intensity"] = panel["oil_consumption_twh"] / (panel["gdp_usd"] / 1e9)
    panel["fx_change_pct"] = g["exchange_rate_lcu_per_usd"].pct_change(fill_method=None) * 100

    # Interaction: price shock hits importers harder
    panel["shock_x_import_burden"] = panel["brent_yoy_pct"] * panel["import_burden_pct_gdp"]

    # Targets: NEXT year's outcome (we predict t+1 from info at t)
    panel["inflation_next"] = g["inflation_pct"].shift(-1)
    panel["gdp_growth_next"] = g["gdp_growth_pct"].shift(-1)
    panel["stress_next"] = np.where(panel["inflation_next"].isna(), np.nan,
                                    (panel["inflation_next"] > INFLATION_STRESS_THRESHOLD).astype(float))
    return panel


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel = build()
    panel.to_csv(PROCESSED / "panel.csv", index=False)

    print(f"Panel shape: {panel.shape}")
    print("\nMissing share by column (%):")
    print((panel.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())
    lab = panel.dropna(subset=["stress_next"])
    print(f"\nRows with a target: {len(lab)}  |  stress rate: {lab['stress_next'].mean():.1%}")
