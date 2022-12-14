# -*- coding: utf-8 -*-
"""
A CLI to quickly check if HDF files have potentially anomalous values. This
will check all files with an 'h5' extension in the current directory.
The checks, so far, only include whether the minimum and maximum values are
within the ranges defined in the "VARIABLE_CHECKS" object.'

Things to do:
    
    1) Add in more threshold values, infer output type, maybe using units

    3) Suggest uniform meta data for all of our resource data sets.
    4) Make it work better.
    5) Remove xml files, address "Cannot open {}.xml" problem.
    
Created on Mon Nov 25 08:52:00 2019

@author: twillia2
"""

import json
import multiprocessing as mp
import os
import sys

from glob import glob

import click
import h5py
import numpy as np
import pandas as pd

from osgeo import gdal
from revruns import VARIABLE_CHECKS
from tqdm import tqdm


# Help printouts
DIR_HELP = "The directory from which to read the HDF5 files (Defaults to '.')."
SAVE_HELP = ("The path to use for the csv output file (defaults to " +
             "'./checkvars.csv').")
EXT_HELP = ("The HDF5 file extension to check. Defaults to 'h5'. (str)")


def gdal_info(file):
    """ gdalinfo is able to retrieve summary statistics of multidimensional
    data sets quickly enough, but it doesn't even detect single dimensional
    data sets in hdf files.
    """
    # GDAL For multidimensional data sets
    try:
        pointer = gdal.Open(file)
        health = "good"

        # Get the list of sub data sets in each file
        subds = pointer.GetSubDatasets()

        # If there was only one detectable data set, use this instead
        if len(subds) == 0:
            subds = [(pointer.GetDescription(),)]
            assert "HDF5" in subds[0][0]

    except RuntimeError:
        print("Broken file: " + os.path.basename(file))
        health = "broken"

    except AssertionError:
        health = "questionable"

    # If this opens
    if health == "good":
        # For each of these sub data sets, get an info dictionary
        stat_dicts = []
        for sub in subds:
            
            # Turn off most options to try and speed this up
            info_str = gdal.Info(sub[0],
                                 stats=True,
                                 showFileList=True,
                                 format="json",
                                 listMDD=False,
                                 approxStats=False,
                                 deserialize=False,
                                 computeMinMax=False,
                                 reportHistograms=False,
                                 reportProj4=False,
                                 computeChecksum=False,
                                 showGCPs=False,
                                 showMetadata=False,
                                 showRAT=False,
                                 showColorTable=False,
                                 allMetadata=False,
                                 extraMDDomains=None)


            # Read this as a dictionary
            info = json.loads(info_str)
            desc = info["description"]
            ds = desc[desc.index("//") + 2: ]  # data set name
            stats = info["bands"][0]

            # Try for thresholds
            try:
                max_threshold = VARIABLE_CHECKS[ds][1]
            except KeyError:
                max_threshold = np.nan
            try:
                min_threshold = VARIABLE_CHECKS[ds][0]
            except KeyError:
                min_threshold = np.nan

            # Return just these elements
            stat_dict = {"file": file,
                         "data_set": ds,
                         "min": stats["minimum"],
                         "max": stats["maximum"],
                         "mean": stats["mean"],
                         "std": stats["stdDev"]}
            stat_dicts.append(stat_dict)

        # Remove xml file
        os.remove(".".join([file, "aux", "xml"]))

        # To better account for completed data sets, make a data frame
        gdal_data = pd.DataFrame(stat_dicts)

    else:
        gdal_data = pd.DataFrame(columns = ["file", "data_set", "min", "max",
                                            "mean", "std"])

    return gdal_data, health


def single_info(file):
    """Return summary statistics of all data sets in a single hdf5 file.

    So, GDAL handles the multi-dimensional data sets fairly quickly, but doesn't
    even detect the singular dimensional data sets. Actually, it doesn't always
    detect the multidimensional data sets!

    file = "/shared-projects/rev/projects/india/forecast/pv/"
    """
    # Get the summary statistics data frame for multidimensional data sets
    gdal_data, health = gdal_info(file)

    # It might be empty
    gdal_datasets = list(gdal_data["data_set"].values)

    # H5py for one-dimensional data sets
    if health == "good":
        with h5py.File(file, "r") as data_set:
            keys = data_set.keys()
            keys = [k for k in keys if k not in ["meta", "time_index",
                                                 "key_reference"]]
            scale_factors = {k: data_set[k].attrs["scale_factor"] for k in keys}
            units = {k: data_set[k].attrs["units"] for k in keys}
            keys = [k for k in keys if k not in gdal_datasets]
            data_sets = {k: data_set[k][:] for k in keys}
            meta = data_set["meta"][:]

        # Now we have to calculate these "manually" (lol)
        stat_dicts = []
        for ds in data_sets.keys():
            values = data_sets[ds]
            if ds in VARIABLE_CHECKS:
                max_threshold = VARIABLE_CHECKS[ds][1]
                min_threshold = VARIABLE_CHECKS[ds][0]
            else:
                max_threshold = np.nan
                min_threshold = np.nan
            minv = np.min(values)
            maxv = np.max(values)
            meanv = np.mean(values)
            stdv = np.std(values)
            stat_dict = {"file": file,
                         "data_set": ds,
                         "min": minv,
                         "max": maxv,
                         "mean": meanv,
                         "std": stdv}
            stat_dicts.append(stat_dict)

        # Make another data frame with the 1-D data set statistics
        hdf_data = pd.DataFrame(stat_dicts)

        # Concatenate the two together if needed
        summary_df = pd.concat([gdal_data, hdf_data]).reset_index(drop=True)

        # Add the scale factors and units
        summary_df["scale_factor"] = summary_df["data_set"].map(scale_factors)
        summary_df["units"] = summary_df["data_set"].map(units)

        # Now let's get a clue to our location
        meta_df = pd.DataFrame(meta)
        meta_df["reV_tech"] = meta_df["reV_tech"].apply(
                lambda x: x.decode("utf-8"))

        # Lets also get overall min and max
        group = summary_df.groupby("data_set")
        summary_df["overall_min"] = group["min"].transform("min")
        summary_df["overall_max"] = group["max"].transform("max")

        # Finally, in case the file is broken, add a flag
        summary_df["file_health"] = health

    else:
        columns = ['file', 'data_set', 'min', 'max', 'mean', 'std',
                   'scale_factor', 'units', 'overall_min', 'overall_max',
                   'file_health']
        values = {c: np.nan for c in columns}
        summary_df = pd.DataFrame(values, index=[0])
        summary_df["file"] = file
        summary_df["file_health"] = health

    return summary_df


# The command
@click.command()
@click.option("--directory", "-d", default=".", help=DIR_HELP)
@click.option("--savepath", "-p", default="./checkvars.csv", help=SAVE_HELP)
@click.option("--extension", "-e", default="h5", help=EXT_HELP)
def main(directory, savepath, extension):
    """
    revruns Check

    Checks all hdf5 files in a current directory for threshold values in
    data sets. This uses GDALINFO and also otputs an XML file with summary
    statistics and attribute information for each hdf5 file.
    """

    # Go to that directory
    os.chdir(directory)

    # Expand paths for the csv
    directory = os.path.expanduser(directory)
    directory = os.path.abspath(directory)
    savepath = os.path.expanduser(savepath)
    savepath = os.path.abspath(savepath)

    # Overwrite existing
    if os.path.exists(savepath):
        os.remove(savepath)

    # Get and open files.
    files = glob(os.path.join(directory, "*" + extension))

    # How many cpus do we have?
    ncores = mp.cpu_count()

    # Try to dl in parallel with progress bar
    with mp.Pool(ncores - 1) as pool:
        info_dfs = []
        for info_df in tqdm(pool.imap(single_info, files), total=len(files),
                            position=0, file=sys.stdout):
            info_dfs.append(info_df)

    # Combine output into single data frame
    info_df = pd.concat(info_dfs)
    info_df = info_df.reset_index(drop=True)

    # Write to file
    info_df.to_csv(savepath, index=False)

if __name__ == "__main__":
    main()
