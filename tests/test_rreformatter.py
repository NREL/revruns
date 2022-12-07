# -*- coding: utf-8 -*-
"""Test rreformatter

Created on Fri Jan 28 16:57:22 2022

@author: twillia2
"""
import os
import tempfile

from revruns import rr
from revruns.rreformatter import Reformatter


TEMPLATE = "data/rasters/pr_template.tif"
INPUT_DICT = {
    "cyclone_names": {
        "path": "data/shapefiles/PR_Tropical_Cyclone_Storm_Segments_32161.geojson",
        "field": "stormName",
        "buffer": 0
    },
    "significant_wave_height_annual": {
        "path": "data/shapefiles/pr_wave_sig_ht_32161.geojson",
        "field": "ann_ssh",
        "buffer": 0
    }
}
INPUT_FPATH = "data/tables/rev_inputs.xlsx"
INPUTS = rr.get_sheet(INPUT_FPATH, "data")


def format_inputs():
    """Format input data frame into rreformatter input dictionary."""
    inputs = {}
    for i, row in INPUTS.iterrows():
        inputs[row["name"]] = {
            "path": row["path"],
            "layer": row["layer"],
            "field": row["field"],
            "description": row["description"],
            "source": row["source"]
        }
    return inputs


def test_dict():
    """Test sample refromatting routine with input dictionary."""
    inputs = format_inputs()

def test_table():
    """Test sample refromatting routine with input excel file."""
    # Initialize object
    with tempfile.TemporaryDirectory() as out_dir:
        excl_fpath = os.path.join(out_dir, "Test_Exclusions.h5")
        inputs = rr.get_sheet(INPUT_FPATH, "data")
        reformatter = Reformatter(
            inputs=inputs,
            out_dir=out_dir,
            template=TEMPLATE,
            excl_fpath=excl_fpath
        )
        reformatter.main()

        # Run reformatting method

        # Assertion tests

        # Cleanup out_dir

