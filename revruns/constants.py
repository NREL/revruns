# -*- coding: utf-8 -*-
"""Constants for revruns.

Created on Wed Jun 24 20:52:25 2020

@author: twillia2
"""
import os

from osgeo import gdal
from reV.cli import commands



# For filtering COUNS
CONUS_FIPS = ["54", "12", "17", "27", "24", "44", "16", "33", "37", "50", "09",
              "10", "35", "06", "34", "55", "41", "31", "42", "53", "22", "13",
              "01", "49", "39", "48", "08", "45", "40", "47", "56", "38", "21",
              "23", "36", "32", "26", "05", "28", "29", "30", "20", "18", "46",
              "25", "51", "11", "19", "04"]

# For package data
ROOT = os.path.abspath(os.path.dirname(__file__))


# For checking if a requested output requires economic treatment.
ECON_MODULES = ["flip_actual_irr",
                "lcoe_nom",
                "lcoe_real",
                "ppa_price",
                "project_return_aftertax_npv"]

# Checks for reasonable model output value ranges. No scaling factors here.
VARIABLE_CHECKS = {
    "poa": (0, 1000),  # 1,000 MW m-2
    "cf_mean": (0, 240),  # 24 %
    "cf_profile": (0, 990),  # 99 %
    "ghi_mean": (0, 1000),
    "lcoe_fcr": (0, 1000)
}

# Resource data set dimensions. Just the number of grid points for the moment.
RESOURCE_DIMS = {
    "nsrdb_v3": 2018392,
    "wind_conus_v1": 2488136,
    "wind_canada_v1": 2894781,
    "wind_canada_v1bc": 2894781,
    "wind_mexico_v1": 1736130,
    "wind_conus_v1_1": 2488136,
    "wind_canada_v1_1": 289478,
    "wind_canada_v1_1bc": 289478,
    "wind_mexico_v1_1": 1736130
}

# The Eagle HPC path to each resource data set. Brackets indicate years.
RESOURCE_DATASETS = {
        "nsrdb_v3": "/datasets/NSRDB/v3/nsrdb_{}.h5",
        "nsrdb_india": "/datasets/NSRDB/india/nsrdb_india_{}.h5",
        "wtk_conus_v1": "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5",
        "wtk_canada_v1": "/datasets/WIND/canada/v1.0.0/wtk_canada_{}.h5",
        "wtk_canada_v1bc": "/datasets/WIND/canada/v1.0.0bc/wtk_canada_{}.h5",
        "wtk_mexico_v1": "/datasets/WIND/mexico/v1.0.0/wtk_mexico_{}.h5",
        "wtk_conus_v1_1": "/datasets/WIND/conus/v1.1.0/wtk_conus_{}.h5",
        "wtk_canada_v1_1": "/datasets/WIND/canada/v1.1.0/wtk_canada_{}.h5",
        "wtk_canada_v1_1bc": ("/datasets/WIND/canada/v1.1.0bc/" +
                                "wtk_canada_{}.h5"),
        "wtk_mexico_v1_1": "/datasets/WIND/mexico/v1.0.0/wtk_mexico_{}.h5"
        }


# The title of each resource data set.
RESOURCE_LABELS = {
        "nsrdb_v3": "National Solar Radiation Database -  v3.0.1",
        "nsrdb_india": "National Solar Radiation Database - India",
        "wtk_conus_v1": ("Wind Integration National Dataset (WIND) " +
                          "Toolkit - CONUS, v1.0.0"),
        "wtk_canada_v1": ("Wind Integration National Dataset (WIND) " +
                           "Toolkit - Canada, v1.0.0"),
        "wtk_canada_v1bc": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Canada, v1.1.0"),
        "wtk_mexico_v1": ("Wind Integration National Dataset (WIND) " +
                           "Toolkit - Mexico, v1.0.0"),
        "wtk_conus_v1_1":("Wind Integration National Dataset (WIND) " +
                           "Toolkit - CONUS, v1.1.0"),
        "wtk_canada_v1_1": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Canada, v1.1.0"),
        "wtk_canada_v1_1bc": ("Wind Integration National Dataset (WIND) " +
                               "Toolkit - Canada, v1.1.0bc"),
        "wtk_mexico_v1_1": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Mexico, v1.0.0"),
        }


PIPELINE_TEMPLATE = {
    "logging": {
        "log_file": None,
        "log_level": "INFO"
    },
    "pipeline": [
        {"generation": "./config_gen.json"},
        {"collect": "./config_collect.json"},
        {"econ": "./config_econ.json"},
        {"multi-year": "./config_multi-year.json"},
        {"supply-curve-aggregation": "./config_aggregation.json"},
        {"supply-curve": "./config_supply-curve.json"},
        {"rep-profiles": "./config_rep-profiles.json"}
    ]
}

TEMPLATES = {}
for command in commands:
    TEMPLATES[command.name] = command.documentation.template_config
TEMPLATES["pipeline"] = PIPELINE_TEMPLATE


# Default SAM model parameters
SOLAR_SAM_PARAMS = {
    "azimuth": "PLACEHOLDER",
    "array_type": "PLACEHOLDER",
    "capital_cost": "PLACEHOLDER",
    "clearsky": "PLACEHOLDER",
    "compute_module": "PLACEHOLDER",
    "dc_ac_ratio": "PLACEHOLDER",
    "fixed_charge_rate": "PLACEHOLDER",
    "fixed_operating_cost": "PLACEHOLDER",
    "gcr": "PLACEHOLDER",
    "inv_eff": "PLACEHOLDER",
    "losses": "PLACEHOLDER",
    "module_type": "PLACEHOLDER",
    "system_capacity": "PLACEHOLDER",
    "tilt": "PLACEHOLDER",
    "variable_operating_cost": "PLACEHOLDER"
}

WIND_SAM_PARAMS = {
        "adjust:constant": 0,
        "capital_cost" : "PLACEHOLDER",
        "fixed_operating_cost" : "PLACEHOLDER",
        "fixed_charge_rate": "PLACEHOLDER",
        "icing_cutoff_temp": "PLACEHOLDER",
        "icing_cutoff_rh": "PLACEHOLDER",
        "low_temp_cutoff": "PLACEHOLDER",
        "system_capacity": "PLACEHOLDER",
        "variable_operating_cost": 0,
        "turb_generic_loss": 16.7,
        "wind_farm_wake_model": 0,
        "wind_farm_xCoordinates": [0],
        "wind_farm_yCoordinates": [0],
        "wind_resource_model_choice": 0,
        "wind_resource_shear": 0.14,
        "wind_resource_turbulence_coeff": 0.1,
        "wind_turbine_cutin": "PLACEHOLDER",  # Isn't this inferred in the pc?
        "wind_turbine_hub_ht": "PLACEHOLDER",
        "wind_turbine_powercurve_powerout": "PLACEHOLDER",
        "wind_turbine_powercurve_windspeeds": "PLACEHOLDER",
        "wind_turbine_rotor_diameter": "PLACEHOLDER"
}

SAM_TEMPLATES = {
    "pvwattsv5": SOLAR_SAM_PARAMS,
    "pvwattsv7": SOLAR_SAM_PARAMS,
    "windpower": WIND_SAM_PARAMS
}

SLURM_TEMPLATE = (
"""#!/bin/bash

#SBATCH --account=PLACEHOLDER
#SBATCH --time=1:00:00
#SBATCH -o PLACEHOLDER.o
#SBATCH -e PLACEHOLDER.e
#SBATCH --job-name=<PLACEHOLDER>
#SBATCH --nodes=1
#SBATCH --mail-user=PLACEHOLDER
#SBATCH --mem=79000

echo Running on: $HOSTNAME, Machine Type: $MACHTYPE
echo CPU: $(cat /proc/cpuinfo | grep "model name" -m 1 | cut -d:  -f2)
echo RAM: $(free -h | grep  "Mem:" | cut -c16-21)

source ~/.bashrc
module load conda
conda activate /path/to/env/

python script.py
"""
)