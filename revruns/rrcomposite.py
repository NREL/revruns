# -*- coding: utf-8 -*-
"""Build a singular composite inclusion layer and write to geotiff.

Created on Wed Feb  9 09:10:26 2022

@author: twillia2
"""
import click
import json
import os

import h5py
import rasterio as rio

from reV.supply_curve.exclusions import ExclusionMaskFromDict
from rex import init_logger


HELP = {
    "config": ("Path to reV aggregation JSON configuration file containing an "
               "exclusion dictionary ('excl_dict')."),
    "dst": ("Destination path to output GeoTiff. Defaults to current directory"
            "using the name of aggregation config file."),
}


def composite(config, dst=None):
    """Use an aggregation config to build a composite inclusion raster.

    Parameters
    ----------
    config: str | dict
        Path to reV aggregation JSON configuration file containing an exclusion
        dictionary ('excl_dict').
    dst : str
        Destination path to output GeoTiff. Defaults to current directory
        using the name of containing directory of the aggregation config file.
    """
    # Create job name and destination path
    if isinstance(config, str):
        config = os.path.abspath(config)
        with open(config, "r") as file:
            conf = json.load(file)
        name = os.path.dirname(config).split("/")[-1]
        if not dst:
            dst = "./" + os.path.dirname(config).split("/")[-1] + ".tif"
    else:
        conf = config
        if not dst:
            raise OSError(f"If the 'config' argument is a dict, a destination"
                          " file path is required.")
        name = os.path.dirname(dst).split("/")[-1]

    init_logger("rrcomposite", log_level="DEBUG", log_file=name + ".log")

    excl_dict = conf["excl_dict"]
    excl_fpath = conf["excl_fpath"]

    # Run composite builder
    _composite(excl_dict, excl_fpath, dst)


def _composite(excl_dict, excl_fpath, dst=None, min_area=None):
    """Use an exclusion dictionary, hdf5 fpath, and destination path."""
    # Run reV to merge
    masker = ExclusionMaskFromDict(excl_fpath, layers_dict=excl_dict,
                                   min_area=min_area)
    mask = masker.mask
    mask = mask.astype("uint8")

    # Get a raster profile from the h5 dataset
    if isinstance(excl_fpath, list):
        template = excl_fpath[0]
    else:
        template = excl_fpath
    with h5py.File(template, "r") as ds:
        profile = json.loads(ds.attrs["profile"])
    profile["dtype"] = str(mask.dtype)
    profile["nodata"] = None
    profile["blockxsize"] = 256
    profile["blockysize"] = 256
    profile["tiled"] = "yes"
    profile["compress"] = "lzw"

    # Save to a single raster
    print(f"Saving output to {dst}...")
    with rio.open(dst, "w", **profile) as file:
        file.write(mask, 1)


@click.command()
@click.option("--config", "-c", required=1, help=HELP["config"])
@click.option("--dst", "-d", default=None, help=HELP["dst"])
def main(config, dst):
    """Combine exclusion layers into one composite inclusion raster."""
    composite(config, dst)


if __name__ == "__main__":
    pass