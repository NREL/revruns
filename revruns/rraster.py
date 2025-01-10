# -*- coding: utf-8 -*-
"""Create a raster out of an HDF point file."""
import click
import os
import shutil
import subprocess as sp

from pathlib import Path

import fiona
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio as rio

from pyproj import CRS
from revruns import rr
from revruns.gdalmethods import rasterize

from scipy.spatial import cKDTree

os.environ['PROJ_NETWORK'] = 'OFF'


FILE_HELP = "The file from which to create the shape geotiff. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current "
             "directory with the basename of the csv/h5 file. (str)")
VARIABLE_HELP = ("For HDF5 files, the data set to rasterize. For CSV files, "
                 "the field to rasterize. (str)")
CRS_HELP = ("Coordinate reference system. Pass as <'authority':'code'>. (str)")
RES_HELP = ("Target resolution, in the same units as the CRS. (numeric)")
AGG_HELP = ("For HDF5 time series, the aggregation function to use to render"
            " the raster layer. Any appropriate numpy method. If 'layer' is"
            " provided, this will be ignored."
            " defaults to mean. (str)")
LAYER_HELP = ("For HDF5 time series, the time index to render. If attempting "
              "to rasterize a time series and this isn't provided, an "
              "aggregation function will be used. Defaults to 0. (int)")
FILTER_HELP = ("A column name, value pair to use to filter the data before "
               "rasterizing (e.g. rraster -f state -f Georgia ...). (list)")
FILL_HELP = ("Fill na values by interpolation. (boolen)")
CUT_HELP = ("Path to vector file to use to clip output. (str)")



def write_raster(grid, transform, crs, dst):
    """Write an array to a geotiff."""
    # Format the CRS
    crs = CRS(crs)

    # Create a rasterio style georeferencing profile
    profile = {
        'driver': 'GTiff',
        'dtype': str(grid.dtype),
        'nodata': None,
        'width': grid.shape[1],
        'height': grid.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'tiled': False,
        'interleave': 'band'
     }

    with rio.Env():
        with rio.open(dst, "w", **profile) as file:
            file.write(grid, 1)


def csv(src, dst, variable, resolution, crs, fillna, cutline):
    # This is inefficient
    df = pd.read_csv(src)
    if "geometry" in df:
        del df["geometry"]
        df = pd.DataFrame(df)
    gdf = df.rr.to_geo()
    gdf = gdf[["geometry", variable]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # And finally rasterize
    grid, transform, gridy, gridx = to_grid(gdf, variable, resolution)

    # And write to raster
    write_raster(grid, transform, crs, dst)


def get_scale(ds, variable):
    attrs = ds[variable].attrs.keys()
    scale_key = [k for k in attrs if "scale_factor" in k][0]
    if scale_key:
        scale = ds[variable].attrs[scale_key]
    else:
        scale = 1
    return scale


def gpkg(src, dst, variable, resolution, crs, fillna, cutline):
    # This is inefficient
    gdf = gpd.read_file(src)
    gdf = gdf[["geometry", variable]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # Convert to true grid
    grid, transform, gridy, gridx = to_grid(gdf, gdf[variable], resolution)

    # And write to raster
    write_raster(grid, transform, crs, dst)


def h5(src, dst, variable, resolution, crs, agg_fun, layer, fltr, fillna,
       cutline):
    """
    Rasterize dataset in HDF5 file.

    Parameters
    ----------
    src : str
        DESCRIPTION.
    dst : str
        DESCRIPTION.
    variable : str
        DESCRIPTION.
    resolution : int
        DESCRIPTION.
    crs : str
        DESCRIPTION.
    agg_fun : str
        DESCRIPTION.
    layer : str
        DESCRIPTION.
    fltr : str
        DESCRIPTION.
    fillna : bool
        DESCRIPTION.
    cutline : str
        DESCRIPTION.

    Returns
    -------
    """
    # Open the file
    with h5py.File(src, "r") as ds:
        meta = pd.DataFrame(ds["meta"][:])
        meta.rr.decode()

        # Is this a time series or single layer
        scale = get_scale(ds, variable)
        if len(ds[variable].shape) > 1:
            data = h5_timeseries(ds, variable, agg_fun, layer)
            field = "{}_{}".format(variable, agg_fun)
        else:
            data = ds[variable][:]
            field = variable

    # Append the data to the meta object and create geodataframe
    meta[field] = data / scale

    # Do we wnt to apply a filter? How to parameterize that?  # <-------------- Not ready
    if fltr:
        meta = meta[meta[fltr[0]] == fltr[1]]

    # Create a GeoDataFrame
    gdf = meta.rr.to_geo()
    gdf = gdf[["geometry", field]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # Create a consistent grid out of this
    gdf = to_grid(gdf, variable, resolution)

    # And finally rasterize
    rrasterize(gdf, resolution, dst, fillna, cutline)

    return dst


def h5_timeseries(ds, dataset, agg_fun, layer):
    # Specifying a layer overrides the aggregation
    if layer:
        data = ds[dataset][layer]
    else:
        fun = getattr(np, agg_fun)
        data = fun(ds[dataset][:], axis=0)
    return data


def rrasterize(gdf, resolution, dst, fillna=False, cutline=None,
               variable=None):
    # Make sure we have the target directory
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    # Not doing this in memory in case it's big
    tmp_src = str(dst).replace(".tif", "_tmp.gpkg")
    gdf.to_file(tmp_src, driver="GPKG")
    layer_name = fiona.listlayers(tmp_src)[0]

    # There will only be two columns
    if not variable:
        variable = gdf.columns[1]

    # Set no data value
    dtype = str(gdf[variable].dtype)
    if "float" in dtype:
        na = np.finfo(dtype).max
    else:
        na = np.iinfo(dtype).max

    # Get extent
    te = [str(b) for b in gdf.total_bounds]

    # Write to dst
    sp.call(
        [
            "gdal_rasterize",
            tmp_src, dst,
            "-l", layer_name,
            "-a", variable,
            "-a_nodata", str(na),
            "-at",
            "-te", *te,
            "-tr", str(resolution), str(-int(resolution))
        ]
    )

    # Fill na values
    if fillna:
        tmp_dst = str(dst).replace(".tif", "_tmp.tif")
        sp.call(["gdal_fillnodata.py", dst, tmp_dst])
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(tmp_dst, dst)

    # Cut to vector
    if cutline:
        cutline = str(Path(cutline).expanduser())
        tmp_dst = str(dst).replace(".tif", "_tmp.tif")
        sp.call(["gdalwarp", dst, tmp_dst, "-crop_to_cutline", "-cutline",
                 cutline])
        os.remove(dst)
        shutil.move(tmp_dst, dst)

    # Get rid of temporary shapefile
    os.remove(tmp_src)


def to_grid(gdf, variable, resolution):
    """Convert coordinates from an irregular point dataset into an even grid.

    Parameters
    ----------
    gdf : geopandas.geodataframe.GeoDataFrame
        A geopandas data frame of coordinates.
    variable : str
        Variable to covert to a grid.
    resolution: int | float
        The resolution of the target grid.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Returns a 3D array (y, x, time) of data values a 2D array of coordinate
        values (nxy, 2).

    Notes
    -----
    - This only takes about a minute for a ~500 X 500 X 8760 dim dataset, but
    it eats up memory. If we saved the df to file, opened it as a dask data
    frame, and generated the arrays as dask arrays we might be able to save
    space.
    """
    # Get the data
    data = gdf[variable].values

    # Get the extent
    minx, miny, maxx, maxy = gdf.total_bounds

    # These are centroids of points, we want top left corners
    resolution = int(resolution)
    minx -= (resolution / 2)
    maxx -= (resolution / 2)
    miny += (resolution / 2)
    maxy += (resolution / 2)

    # Estimate target grid coordinates
    gridx = np.arange(minx, maxx, resolution)
    gridy = np.arange(maxy, miny, -resolution)
    grid_points = np.array(np.meshgrid(gridy, gridx)).T.reshape(-1, 2)

    # Go ahead and make the geotransform
    geotransform = [resolution, 0, minx, 0, -resolution, maxy]

    # Get source point coordinates
    gdf["x"] = gdf["geometry"].apply(lambda p: p.x - (resolution / 2))
    gdf["y"] = gdf["geometry"].apply(lambda p: p.y + (resolution / 2))
    points = gdf[["y", "x"]].values

    # Build kdtree  # <-------------------------------------------------------- Nearest neighbor is not appropriate for the irregular grids like the WTK, what to do? Interpolate here?
    ktree = cKDTree(grid_points)
    _, indices = ktree.query(points)

    # Those indices associate grid point coordinates with the original points
    gdf["gy"] = grid_points[indices, 0]
    gdf["gx"] = grid_points[indices, 1]

    # And these indices indicate the 2D cartesion positions of the grid
    gdf["iy"] = gdf["gy"].apply(lambda y: np.where(gridy == y)[0][0])
    gdf["ix"] = gdf["gx"].apply(lambda x: np.where(gridx == x)[0][0])

    # Okay, now use this to create our empty target grid and assign values
    if len(data.shape) == 1:
        grid = np.zeros((gridy.shape[0], gridx.shape[0]))
        grid[grid == 0] = np.nan
        grid[gdf["iy"].values, gdf["ix"].values] = data
    else:
        grid = np.zeros((data.shape[0], gridy.shape[0], gridx.shape[0]))
        grid[grid == 0] = np.nan
        grid[:, gdf["iy"].values, gdf["ix"].values] = data

    # Go ahead and return the x and y coordinates. Calculating from geom
    # can lead to data precision errors

    return grid, geotransform, gridy, gridx


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--variable", "-v", required=True, help=VARIABLE_HELP)
@click.option("--resolution", "-r", required=True, help=RES_HELP)
@click.option("--crs", "-c", default="epsg:4326", help=CRS_HELP)
@click.option("--agg_fun", "-a", default="mean", help=AGG_HELP)
@click.option("--layer", "-l", default=None, help=LAYER_HELP)
@click.option("--fltr", "-f", default=None, multiple=True, help=FILTER_HELP)
@click.option("--fillna", "-fn", is_flag=True, help=FILL_HELP)
@click.option("--cutline", "-cl", default=None, help=CUT_HELP)
def main(src, dst, variable, resolution, crs, agg_fun, layer, fltr, fillna,
         cutline):
    """REVRUNS - RRASTER - Rasterize a reV output."""
    # Get the file extension and call the appropriate function
    extension = os.path.splitext(src)[-1]
    if extension == ".h5":
        h5(src, dst, variable, resolution, crs, agg_fun, layer, fltr, fillna,
           cutline)
    elif extension == ".csv":
        csv(src, dst, variable, resolution, crs, fillna, cutline)
    elif extension == ".gpkg":
        gpkg(src, dst, variable, resolution, crs, fillna, cutline)


# if __name__ == "__main__":
#     resolution = 2_500
#     crs = "esri:102008"
#     variable = "cf_mean"
#     cutline = None
#     src = "/Users/twillia2/transfer/geotherm_gen_sample.csv"
#     dst = "/Users/twillia2/transfer/geotherm_gen_sample.tif"
#     dst = os.path.expanduser(dst)
#     fillna = True
#     if os.path.exists(dst):
#         os.remove(dst)
#     csv(src, dst, variable, resolution, crs, fillna, cutline)
