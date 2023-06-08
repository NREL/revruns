# -*- coding: utf-8 -*-
"""revruns ATB

Methods for accessing ATB values and updating versions.

There are two types of tables, the worksheets and the flat table. The only
problem with the flat table is that it was missing Overnight Capital Costs
last time I checked. However, depending on where you get your FCR, the regular
capital costs might be more appropriate because of the construction period
financing term.

Ideally we have a consistent URL or an API, let's see what they've managed to
work in since i last checked.

Author: twillia2
Date: Fri Apr 15 09:40:28 MDT 2022
"""
from functools import lru_cache

import pandas as pd

from revruns import Paths


TECHNOLOGIES = {
    "battery_commerical": "Commercial Battery Storage",
    "battery_residential": "Residential Battery Storage",
    "battery_utility": "Utility-Scale Battery Storage",
    "biopower": "Biopower",
    "coal": "Coal",
    "csp": "CSP",
    "geothermal": "Geothermal",
    "hydropower": "Hydropower",
    "natural_gas": "Natural Gas",
    "nuclear": "Nuclear",
    "pumped_hydro": "Pumped Storage Hydropower",
    "pv_battery_utility": "Utility-Scale PV-Plus-Battery",
    "pv_commercial": "Commercial PV",
    "pv_residential": "Residential PV",
    "pv_utility": "Utility PV",
    "wind_distributed": "Distributed Wind",
    "wind_distributed_commercial": "Commercial DW",
    "wind_distributed_large": "Large DW",
    "wind_distributed_mid": "Midsize DW",
    "wind_distributed_residential": "Residential DW",
    "wind_onshore_utility": "Land-based Wind",
    "wind_offshore_utility": "OffShore Wind",
}
CASES = {
    "randd": "R&D",
    "market": "Market"
}


class ATB:
    """Methods for retrieving data from the Annual Technology Baseline."""

    def __init__(self, table_fpath=None, atb_year=2022, case="randd",
                 cost_year=2030, lifetime=30, scenario="moderate",
                 tech="wind_onshore_utility"):
        """Initialize ATB object."""
        self.table_fpath = table_fpath
        self.atb_year = atb_year
        self.case = case
        self.cost_year = cost_year
        self.lifetime = lifetime
        self.scenario = scenario
        self.tech = tech
        self.home = Paths.data
        self.host = "https://oedi-data-lake.s3.amazonaws.com"
        self.url = f"{self.host}/ATB/electricity/csv/{atb_year}/ATBe.csv"

    def __repr__(self):
        """Return ATB representation string."""
        msgs = [f"{k}={v}" for k, v in self.__dict__.items()]
        msg = ", ".join(msgs)
        return f"<ATB object: {msg}>"

    @property
    def local_path(self):
        """Return the package data path for a stored table."""
        lpath = self.home.joinpath("atb", str(self.atb_year), "ATBe.csv")
        lpath.parent.mkdir(exist_ok=True, parents=True)
        return lpath

    @property
    @lru_cache
    def full_data(self):
        """Return the full dataset."""
        # Read the full dataset 
        if not self.table_fpath:
            if not self.local_path.exists():
                df = pd.read_csv(self.url, low_memory=False)
                if "Unnamed: 0" in df:
                    del df["Unnamed: 0"]
                df.to_csv(self.local_path, index=False)
            else:
                df = pd.read_csv(self.local_path, low_memory=False)
        else:
            df = pd.read_csv(self.table_fpath, low_memory=False)
        return df

    @property
    def data(self):
        """Return the filtered dataset."""
        df = self.full_data
        df = df[df["technology_alias"] == TECHNOLOGIES[self.tech]]
        df = df[df["core_metric_variable"] == self.cost_year]
        df = df[df["scenario"] == self.scenario.capitalize()]
        df = df[df["core_metric_case"] == CASES[self.case]]
        df = df[df["crpyears"] == str(self.lifetime)]
        df.reset_index(drop=True, inplace=True)
        return df

    @classmethod
    @property
    def technologies(cls):
        """Return lookup of technology - ATB names."""
        return TECHNOLOGIES

    @property
    def capex(self):
        """Return capex for given tech and year."""
        df = self.data
        df = df[df["core_metric_parameter"] == "CAPEX"]
        capex = df["value"].iloc[0]
        return capex

    @property
    def opex(self):
        """Return opex for given tech and year."""




if __name__ == "__main__":
    self = ATB(tech="pv_utility", atb_year=2022, cost_year=2030)
