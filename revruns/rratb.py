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
    "wind_onshore_utility": "Land-Based Wind",
    "wind_offshore_utility": "OffShore Wind",
}
CASES = {
    "randd": "R&D",
    "market": "Market"
}
TURBINE_CLASSES = {  # Keys represent top of ws bin @ 110 m/s
    100: "T1",
    7.1: "T2",
    6.5: "T3",
    5.9: "T4"
}
SCENARIO_CONVERSIONS = {  # Old ATB's have different scenario names
    "Mid": "Moderate",
    "Low": "Advanced",
    "High": "Conservative",
    "Moderate": "Moderate",
    "Advanced": "Advanced",
    "Conservative": "Conservative",
    "*": "*"
}



class ATB:
    """Methods for retrieving data from the Annual Technology Baseline."""

    def __init__(self, atb_year=2023, case="randd",
                 cost_year=2030, lifetime=30, scenario="moderate",
                 tech="wind_onshore_utility"):
        """Initialize ATB object."""
        try:
            assert tech in self.technologies
        except Exception as exc:
            msg = (f"`tech` argument '{tech}' not recognized, check available"
                    " technology keys using ATB.technologies")
            raise AssertionError(msg) from exc

        self.atb_year = atb_year
        self.case = case
        self.cost_year = cost_year
        self.lifetime = lifetime
        self.scenario = scenario
        self.tech = tech
        self.tech_test = tech
        self.data_home = Paths.data
        self.host = "https://oedi-data-lake.s3.amazonaws.com"
        self.url = f"{self.host}/ATB/electricity/csv/{atb_year}/ATBe.csv"
        tech_alias = TECHNOLOGIES[self.tech]
        tech_alias = tech_alias.replace("-", "").replace(" ", "").lower()
        self.tech_alias = tech_alias

    def __repr__(self):
        """Return ATB representation string."""
        address = hex(id(self))
        msgs = [f"\n   {k}={v}" for k, v in self.__dict__.items()]
        msg = ", ".join(msgs)
        return f"<ATB object at {address}>: {msg}"

    def capex(self, res_class=None, tech=None):
        """Return capex for given tech and year.

        Parameters
        ----------
        res_class : int
            Resource class number from 1 to 10. Defaults to class 1.
        tech : int
            Technology subclass selection number. New for wind in 2023 since
            there are now several different turbines that are appropriate for
            different resource classes. Ranges from 1 to 4.

        Returns
        -------
        float : Value representing capital cost for the given year, technology,
                resource class, scenario, and technology sub-class.
        """
        df = self.data.copy()
        df = df[df["core_metric_parameter"] == "CAPEX"]
        df = self._filter(df, res_class=res_class, tech=tech)
        return df["value"]

    def capex_overnight(self, res_class=None, tech=None):
        """Return overnight capex for given tech and year.

        Parameters
        ----------
        res_class : int
            Resource class number from 1 to 10. Defaults to class 1.
        tech : int
            Technology subclass selection number. New for wind in 2023 since
            there are now several different turbines that are appropriate for
            different resource classes. Ranges from 1 to 4.

        Returns
        -------
        float : Value representing capital cost for the given year, technology,
                resource class, scenario, and technology sub-class.
        """
        # Read in data associated with this tech and year
        df = self.data.copy()

        # Older ATB releases don't include Overnight Capital Costs
        if "OCC" in df["core_metric_parameter"]:
            df = df[df["core_metric_parameter"] == "OCC"]
        else:
            raise NotImplementedError(f"No overnight captial cost available "
                                      f"in the {self.atb_year} ATB.")
            # Assuming 1 year of construction
            # pattern = "Interest During Construction"
            # idc = df[df["core_metric_parameter"].str.contains(pattern)]

        df = self._filter(df, res_class=res_class, tech=tech)
        return df["value"]

    def cf_multipliers(self, tech_scenario="moderate", baseline_year=2023,
                       resource_class=5):
        """Build vector of generation multipliers based on a baseline year.

        Parameters
        ----------
        tech_scenario : str
            Target technology advancement scenario.
        baseline_year : int
            The baseline year to use when calculate improvement ratios.
        resource_class : int
            The target resource class. Note, for older ATBs this will refer to
            technical resource groups rather than resource classes.

        Returns
        -------
        dict : A dictionary of year improvement ratio key-value pairs.
        """
        # Get the full data table, everything else is pre-filtered
        df = self.full_data

        # Standardize technology aliases
        df[self.tech_field] = df[self.tech_field].apply(
            lambda x: x.replace("-", "").replace(" ", "").lower()
        )

        # Standardize scenario names
        df.loc[df["scenario"] == "Constant", "scenario"] = "Conservative"
        df.loc[df["scenario"] == "Mid", "scenario"] = "Moderate"
        df.loc[df["scenario"] == "Low", "scenario"] = "Advanced"

        # Filter
        cdf = df[
            (df["core_metric_parameter"] == "CF") &
            (df[self.tech_field] == self.tech_alias) &
            (df["techdetail"].str.endswith(f"{resource_class}")) &
            (df["scenario"] == tech_scenario.capitalize())
        ]
        cdf = cdf[["core_metric_variable", "core_metric_parameter", "value"]]
        cdf = cdf.drop_duplicates()

        # Calculate multipliers
        base = cdf["value"][cdf["core_metric_variable"] == baseline_year]
        base = base.iloc[0]
        cdf["mult"] = cdf["value"] / base

        # Return as simplified object
        cf_mults = dict(zip(cdf["core_metric_variable"], cdf["mult"]))

        return cf_mults

    def confin(self, res_class=None, tech=None):
        """Return construction financing factor for given tech and year.

        Parameters
        ----------
        res_class : int
            Resource class number from 1 to 10. Defaults to class 1.
        tech : int
            Technology subclass selection number. New for wind in 2023 since
            there are now several different turbines that are appropriate for
            different resource classes. Ranges from 1 to 4.

        Returns
        -------
        float : Value representing construction finance factor for the given
                year, technology, resource class, scenario, and technology
                sub-class.
        """
        df = self.data.copy()
        param = "Interest During Construction  - Nominal"
        df = df[df["core_metric_parameter"] == param]
        df = self._filter(df, res_class=res_class, tech=tech)
        return df["value"]

    @property
    def data(self):
        """Return the filtered dataset."""
        # Copy original data frame
        df = self.full_data.copy()

        # Standardize tech fields
        df.loc[df[self.tech_field].isnull()] = "nan"
        df[self.tech_field] = df[self.tech_field].apply(
            lambda x: x.replace("-", "").replace(" ", "").lower()
        )

        # Filter for technology and cost year
        df = df[df[self.tech_field] == self.tech_alias]
        df = df[df["core_metric_variable"] == self.cost_year]

        # Account for scenario naming discrepancies
        df["scenario"] = df["scenario"].map(SCENARIO_CONVERSIONS)
        df = df[df["scenario"].isin([self.scenario.capitalize(), "*"])]
        df = df[df["core_metric_case"] == CASES[self.case]]
        df = df[df["crpyears"].astype(str) == str(self.lifetime)]
        df.reset_index(drop=True, inplace=True)

        return df

    @property
    def local_path(self):
        """Return the package data path for a stored table."""
        lpath = self.data_home.joinpath("atb", str(self.atb_year), "ATBe.csv")
        lpath.parent.mkdir(exist_ok=True, parents=True)
        return lpath

    @property
    @lru_cache
    def full_data(self):
        """Return the full dataset."""
        if not self.local_path.exists():
            df = pd.read_csv(self.url, low_memory=False)
            if "Unnamed: 0" in df:
                del df["Unnamed: 0"]
            df.to_csv(self.local_path, index=False)
        else:
            df = pd.read_csv(self.local_path, low_memory=False)

        return df

    def opex(self, res_class=None, tech=None):
        """Return opex for given tech and year.

        Parameters
        ----------
        res_class : int
            Resource class number from 1 to 10. Defaults to class 1.
        tech : int
            Technology subclass selection number. New for wind in 2023 since
            there are now several different turbines that are appropriate for
            different resource classes. Ranges from 1 to 4.

        Returns
        -------
        float : Value representing annual operating costs for the given
                year, technology, resource class, scenario, and technology
                sub-class.
        """
        df = self.data.copy()
        df = df[df["core_metric_parameter"] == "Fixed O&M"]
        df = self._filter(df, res_class=res_class, tech=tech)
        return df["value"]

    @classmethod
    @property
    def technologies(cls):
        """Return lookup of technology - ATB names."""
        return TECHNOLOGIES

    @property
    def tech_field(self):
        """Return tech field for given ATB year."""
        tech_field = "technology_alias"
        if "technology_alias" not in self.full_data:
            tech_field = "technology"
        return tech_field

    def _filter(self, df, tech=None, res_class=None):
        if not res_class and not tech:
            df = df[df["techdetail"].str.endswith("1")].iloc[0]
        if res_class:
            df = df[df["techdetail"] .str.endswith(f"{res_class}")].iloc[0]
        if tech:
            df = df[df["techdetail2"].str.contains(f"{tech}")].iloc[0]
        return df
