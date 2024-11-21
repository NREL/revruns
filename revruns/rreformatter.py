# -*- coding: utf-8 -*-
"""Reformat a shapefile or raster into a reV raster using a template.

Note that rasterize partial was adjusted from:
    https://gist.github.com/perrygeo

@date: Tue Apr 27 16:47:10 2021
@author: twillia2
"""
import datetime as dt
import json
import os
import psutil
import shutil
import subprocess as sp
import time
import warnings

from functools import cached_property
from pathlib import Path, PosixPath

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import pyproj
import rasterio as rio

from osgeo_utils import gdal_calc
from pyogrio.errors import DataSourceError
from pyproj import CRS
from rasterio import features
from rasterio.merge import merge
from rtree import index
from shapely import geometry
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from tqdm.auto import tqdm

from revruns.gdalmethods import warp
from revruns.rr import crs_match, isint, isfloat

pyproj.network.set_network_enabled(False)  # For VPN issues
warnings.filterwarnings("ignore", category=FutureWarning)


NEEDED_FIELDS = ["name", "path"]


def fopen():
    return psutil.Process().open_files()


class Rasterizer:
    """Methods for rasterizing vectors."""

    def __init__(self, src, flip=False, resolution=90, crs="esri:102008",
                 template=None, temp_dir=None, chunk_size=10_000,
                 multiprocess=False):
        """Initialize Rasterize object.

        Parameters
        ----------
        src : str | list | gpd.geodataframe.GeoDataFrame
            Path to source vector dataset file or list to such paths or a
            geodataframe.
        template : str
            Path to template raster used for grid geometry. Optional.
        temp_dir : str
            Path to directory to store temporary raster and parquet files.
        flip : boolean
            Reverse values such that 1 represent cells outside of vector and
            0 represent within.
        chunk_size : int
            Chunk side size in meters to use to split `df` for multiprocessing.
        multiprocess : boolean
            Run in parallel on all available cores. This may only be used if
            `src` is a string pointing to a singular file. Defaults to False.
        """
        self.src = Path(src)
        self.template = template
        self.temp_dir = temp_dir
        self.resolution = resolution
        self.crs = crs
        self.flip = flip
        self.chunk_size = chunk_size
        self.multiprocess = multiprocess

        # Set georeferencing, read and process the source data
        self._preflight()

    def __repr__(self):
        """Return Rasterizer representation string."""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() if k != "inputs"]
        pattrs = ",\n  ".join(attrs)
        return f"<Rasterizer object:\n  {pattrs}>"

    def _bounds(self, geom_list):
        """Return the bounds of a geometry."""
        # Get the bounds around all geometries
        bounds = np.array([g.bounds for g in geom_list])

        # Break out the coordinates
        xmin = bounds[:, 0].min()
        ymin = bounds[:, 1].min()
        xmax = bounds[:, 2].max()
        ymax = bounds[:, 3].max()

        # If we have a template, align with that grid
        if self.template:
            with rio.open(self.template, "r") as r:
                transform = r.profile["transform"]
                xr, _, txmin, _, yr, tymax, _, _, _ = list(transform)
                xs = [txmin + xr * i for i in range(r.width)]
                ys = [tymax + yr * i for i in range(r.height)]
            xmin = xs[np.abs(xs - xmin).argmin()] - self.resolution
            xmax = xs[np.abs(xs - xmax).argmin()] + self.resolution
            ymin = ys[np.abs(ys - ymin).argmin()] - self.resolution
            ymax = ys[np.abs(ys - ymax).argmin()] + self.resolution

        return xmin, ymin, xmax, ymax

    def build_profile(self, transform, array):
        """Build a Rasterio-style georeferecing dictionary."""
        if not self.template:
            profile = {
                "transform": transform,
                "height": array.shape[0],
                "width": array.shape[1],
                "count": 1,
                "crs": self.crs,
                "driver": "GTiff",
                "dtype": array.dtype,
                "nodata": None,
                "tiled": False
            }
        else:
            with rio.open(self.template) as r:
                profile = r.profile
            profile["dtype"] = str(array.dtype)

        return profile

    def _clear_temp(self):
        """Delete temporary files."""
        files = list(self.temp_dir.glob("*tif"))
        if files:
            for file in files:
                os.remove(file)

    def _consolidate_vectors(self, fpaths):
        """Combine multiple vector files into one geodataframe."""
        dfs = [gpd.read_parquet(fpath) for fpath in fpaths]
        df = pd.concat(dfs)
        return df

    def _collect_geometries(self, geoseries):
        """Collect and index geometries for a pandas group object."""
        i = 0
        geom_list = []
        ridx = index.Index()
        for geom in geoseries:
            if isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    ridx.insert(i, part.bounds)
                    geom_list.append(part)
                    i += 1
            else:
                ridx.insert(i, geom.bounds)
                geom_list.append(geom)
                i += 1
        return np.array(geom_list), ridx

    @cached_property
    def data(self):
        """Read in or process a source file."""
        # Read data with appropriate method
        if isinstance(self.src, (str, PosixPath)):
            try:
                print(f"Reading in {self.src}...")
                df = gpd.read_file(self.src)
            except DataSourceError:
                df = gpd.read_parquet(self.src)
        elif isinstance(self.src, (list, tuple)):
            print(f"Reading and consolidating {len(self.src)} files...")
            df = self._consolidate_vectors(self.src)
        else:
            df = self.src

        # Set crs and project if needed
        print("Setting coordinate reference information...")
        if not self.crs:
            self.crs = df.crs
        if CRS(self.crs).to_wkt() != df.crs.to_wkt():
            df = df.to_crs(self.crs)

        # Break out multi-polygons
        df = df.explode()

        return df

    def _exterior_ratio(self, geoms, transform, partial, ridx, idx):
        """Calculate the coverage ratio for exterior cells of an array.

        Parameters
        ----------
        geom : list
            List of shapely Polygons.
        transform  : affine.Affine
            Geo transformation represented by the partial grid.
        partial : np.ndarray
            A numpy array to contain the gridded form of the geometry
            vectors.
        ridx : rtree.index.Index
            An R-Tree index containing extent information of each geometry.
        idx : tuple
            A two-item numpy array representing the y- and x-index positions
            that require partial rasterization (i.e., cells along the
            boundary of the geometries).

        Returns
        -------
        np.ndarray : A numpy array containing partial rasterization values.
        """
        # Loop through each set of indices and calculate percent overlap
        idxs = list(zip(*idx))
        for ix in idxs:
            partial = self._ratio(ix, partial, transform, ridx, geoms)

        return partial

    def _build_inputs(self, vector_dst):
        """Build initial inputs for the partial rasterization routine.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            A pandas data frame of features to rasterize.
        vector_dst : str | pathlib.PosixPath
            Path to temporary vector dataset containing all pre-processed
            geometries. Pre-processing includes unpacking multipolygons into
            polygons, combining geometries from multiple files, and assigning
            a grid ID after chunking geometries for parallel processing. This
            will automatically be saved to a geoparquet, regardless of the
            filename extension.

        Returns
        -------
        list : A list of argument values for `_rasterize` or
            `_rasterize_partial`.
        """
        # Split dataframe into spatially neighboring cells, collect indices
        print("Building spatial partitioning grid...")
        vector_dst = vector_dst.parent.joinpath(vector_dst.stem + ".parquet")
        if not vector_dst.exists():
            df = self.grid_chunks(self.data, chunk_size=self.chunk_size)
            df.to_parquet(vector_dst)
        else:
            df = gpd.read_parquet(vector_dst)

        # Collect geometry, build r-tree, unpack multi-polygons
        print("Collecting geometry indices...")
        inputs = []
        for gid in df["grid"].unique():
            entry = {"fpath": vector_dst, "gid": gid}
            inputs.append(entry)
        del df

        return inputs

    def grid_chunks(self, df, chunk_size=10_000):
        """Create a grid to split a geodataframe into roughly `nchunks` parts.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            A pandas data frame of features to rasterize.
        chunk_size : int
            Chunk side size in meters to use to split `df` for multiprocessing.

        Returns
        -------
        pd.core.frame.DataFrame : A data frame split into chunks according to the
            `chunk_size` argument.
        """
        # Polygon Size
        xmin, ymin, xmax, ymax = df.total_bounds
        width = xmax - xmin
        height = ymax - ymin

        # Needed number of cells in each direction
        nx = np.ceil(width / chunk_size)
        ny = np.ceil(height / chunk_size)

        # Each cell distance
        ysize = height / ny
        xsize = width / nx

        # Build grid geometry
        x = xmin
        y = ymin
        array = []
        while y <= ymax:
            while x <= xmax:
                geom = geometry.Polygon([
                    (x, y),
                    (x, y + ysize),
                    (x + xsize, y + ysize),
                    (x + xsize, y),
                    (x, y)
                ])
                array.append(geom)
                x += xsize
            x = xmin
            y += ysize

        # Build grid geodataframe
        grid = gpd.GeoDataFrame(array, columns=["geometry"], crs=df.crs)
        grid["geometry"] = grid["geometry"].buffer(self.resolution * 2)
        grid["grid"] = grid.index

        # Cut the input geometries by this grid
        df = gpd.overlay(df, grid, how="intersection")

        # Sort by grid and reindex so we read using slices
        df = df.sort_values("grid")
        df = df.reset_index()

        return df

    def _preflight(self):
        """Read in source data, set georeferencing, process data."""
        # Read in template information
        if self.template:
            print("Reading template information...")
            with rio.open(self.template) as r:
                profile = r.profile
                self.crs = CRS(profile["crs"])

    def _rasterize(self, geom, shape, transform, exterior=False):
        """Rasterize a single geometry."""
        # Build shapes and account for multipolygons
        if isinstance(geom, geometry.MultiPolygon):
            if exterior:
                shapes = [(g.exterior, 1) for g in geom.geoms]
            else:
                shapes = [(g, 1) for g in geom.geoms]
        else:
            if exterior:
                shapes = [(g.exterior, 1) for g in geom]
            else:
                shapes = [(g, 1) for g in geom]

        # Build array
        array = rio.features.rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True
        )

        return array

    def rasterize_partial(self, dst):
        """Rasterize full vector dataset using percent coverage.

        Parameters
        ----------
        dst : str | pathlib.PosixPath
            Path to destination GeoTiff file.
        """
        # Read in the file and clear the temporary
        start = time.time()
        dst = Path(dst)
        if not dst.name.endswith(".tif"):
            raise OSError(f"Cannot write outputs to {dst}, GeoTiff format "
                          "required.")
        if not self.temp_dir:
            self.temp_dir = dst.parent.joinpath("tmp")
        self.temp_dir.mkdir(exist_ok=True)

        # Chunk data frame and return argument list
        vector_dst = self.temp_dir.joinpath(
            dst.name.replace(".tif", ".parquet")
        )
        input_list = self._build_inputs(vector_dst)

        # Run partial rasterization
        tmps = []
        if self.multiprocess:
            n = mp.cpu_count()
            print(f"Performing partial rasterization with {n} cores...")
            with mp.ThreadPool(n) as pool:
                for tmp in tqdm(pool.imap(self._rasterize_partial, input_list),
                                total=len(input_list)):
                    tmps.append(tmp)
        else:
            print("Performing partial rasterization with 1 core...")
            for inputs in tqdm(input_list):
                tmps.append(self._rasterize_partial(inputs))
        tmps = [str(tmp) for tmp in tmps if tmp]
        print(f"Finished partial rasterization.")

        # Merge individual temporary rasters (python method hangs with SLURM)
        print("Merging rasterized arrays...")
        res = str(self.resolution)
        cmd = ["rio", "merge", "--method=max",  "-r",  res, *tmps,
               f"--output={str(dst)}", "--co", "CHECK_DISK_FREE_SPACE=NO"]
        out = sp.run(cmd, capture_output=True)
        stderr = out.stderr.decode()
        assert out.returncode == 0, f"Error merging rasters: {stderr}"

        # Flip values if requested
        if self.flip:
            print("Flipping values...")
            flip_dst = dst.parent.joinpath(dst.stem + "_flip.tif")
            gdal_calc.Calc("(A - 1) * -1", A=dst, outfile=flip_dst)
            shutil.move(flip_dst, dst)

        # # Write to GeoTiff
        # print(f"Writing output to {dst}...")
        # profile = self.build_profile(transform=transform, array=array)
        # with rio.open(dst, "w", **profile) as file:
        #     file.write(array, 1)

        # Close and remove temporary files
        print("Removing temporary files...")
        for tmp in tmps:
            os.remove(tmp)

        # Print runtime
        end = time.time()
        duration = round((end - start) / 60, 2)
        print(f"Finished, {duration} minutes")

    def _rasterize_partial(self, inputs):
        """Rasterize geometry with percent coverage for partial cells."""
        # Read in inputs
        try:
            fpath = inputs["fpath"]
            gid = inputs["gid"]

            # Create destination file, only run if needed
            tmp = f"{Path(fpath.stem)}_{gid:04d}"
            dst = Path(self.temp_dir.joinpath(tmp + ".tif"))
            if not dst.exists():

                # Read in this grid chunk of the vector
                df = gpd.read_parquet(fpath, filters=[("grid", "==", gid)])
                geoms, ridx = self._collect_geometries(df["geometry"])

                # Build georefencing information
                xmin, ymin, xmax, ymax = self._bounds(geoms)
                height, width = self._shape(geoms)
                shape = (height, width)
                transform = rio.transform.from_bounds(
                    xmin, ymin, xmax, ymax, width, height
                )

                # Create full and boundary arrays (inside and outline of shape)
                full = self._rasterize(geoms, shape, transform)
                boundary = self._rasterize(geoms, shape, transform,
                                           exterior=True)
                partial = (full - boundary).astype("float32")

                # Find the index position of the boundary cells
                idx = np.where(boundary == 1)

                # Loop through indicies of all exterior cells and calc ratio
                partial = self._exterior_ratio(geoms, transform, partial, ridx,
                                               idx)

                # Build a georeferencing dictionary
                profile = {
                    "transform": transform,
                    "height": partial.shape[0],
                    "width": partial.shape[1],
                    "count": 1,
                    "crs": self.crs,
                    "driver": "GTiff",
                    "dtype": partial.dtype,
                    "nodata": None,
                    "tiled": False
                }

                # Save to temporary file
                if partial.shape != (1, 1):
                    with rio.open(dst, "w", **profile) as file:
                        file.write(partial, 1)
                else:
                    dst = None

        except Exception as e:
            print(f"Exception on {dst}: {e}.")
            raise

        return dst

    def _ratio(self, ix, partial, transform, ridx, geoms):
        """Calculate the ratio of geometry coverage within a cell."""
        row_min, col_min = ix
        row_max = row_min + 1
        col_max = col_min + 1
        try:
            # Find cell bounds
            x_min, y_min = transform * (col_min, row_max)
            x_max, y_max = transform * (col_max, row_min)
            bounds = x_min, y_min, x_max, y_max

            # Check if any geometries intersect using the R-Tree index
            keepers = [r for r in ridx.intersection(bounds)]
            cell_geoms = geoms[keepers]
            cell = geometry.box(*bounds)
            overlap = cell.intersection(cell_geoms)
            overlap = [g for g in overlap if not g.is_empty]

            # Construct shapely geometry of cell and intersect geometry
            if overlap:
                if len(overlap) > 1:
                    overlap = unary_union(overlap)
                else:
                    overlap = overlap[0]
                ratio = (overlap.area / cell.area)
                partial[row_min, col_min] = ratio
        except Exception as e:
            print(e)
            raise

        return partial

    def _shape(self, geom):
        """Return target shape for geometry."""
        xmin, ymin, xmax, ymax = self._bounds(geom)
        width = int(np.ceil((xmax - xmin) / self.resolution))
        height = int(np.ceil((ymax - ymin) / self.resolution))
        return (height, width)


class Exclusions:
    """Build or add to an HDF5 Exclusions file."""

    def __init__(self, excl_fpath, attrs=None, lookup={}, parallel=True):
        """Initialize Exclusions object.

        Parameters
        ----------
            excl_fpath : str
                Path to target HDF5 reV exclusion file.
            lookup : str | dict
                Dictionary or path dictionary of raster value, key pairs
                derived from shapefiles containing string values (optional).
            parallel : boolean
                Run GDAL processing routine in parallel with N - 1 cores.
        """
        self.excl_fpath = excl_fpath
        self.attrs = attrs
        self.lookup = lookup
        self.parallel = parallel
        self._preflight()
        self._initialize_h5()

    def __repr__(self):
        """Print the object representation string."""
        msg = "<Exclusions Object:  excl_fpath={}>".format(self.excl_fpath)
        return msg

    def add_layer(self, dname, file, description=None, overwrite=False):
        """Add a raster file and its description to the HDF5 exclusion file."""
        # Open raster object
        raster = rio.open(file)

        # Get profile information
        profile = raster.profile
        profile["crs"] = profile["crs"].to_proj4()
        dtype = profile["dtype"]
        profile = dict(profile)

        # We need a 6 element geotransform, sometimes we receive three extra
        profile["transform"] = profile["transform"][:6]

        # Add coordinates and else check that the new file matches everything
        self._set_coords(profile)
        self._check_dims(file)

        # Add everything to target exclusion HDF
        array = raster.read()
        with h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            if dname in keys:
                if overwrite:
                    del hdf[dname]
                    keys.remove(dname)

            if dname not in keys:
                hdf.create_dataset(name=dname, data=array, dtype=dtype,
                                   chunks=(1, 128, 128))
                hdf[dname].attrs["file"] = os.path.abspath(file)
                hdf[dname].attrs["profile"] = json.dumps(profile)
                if description:
                    hdf[dname].attrs["description"] = description

            if dname in self.lookup:
                string_value = json.dumps(self.lookup[dname])
                hdf[dname].attrs["lookup"] = string_value

    def add_layers(self, file_dict, desc_dict=None, overwrite=False):
        """Add multiple raster files and their descriptions."""
        # Make copies of these dictionaries?
        file_dict = file_dict.copy()
        if desc_dict:
            desc_dict = desc_dict.copy()

        # If descriptions are provided make sure they match the files
        if desc_dict:
            try:
                dninf = [k for k in desc_dict if k not in file_dict]
                fnind = [k for k in file_dict if k not in desc_dict]
                assert not dninf
                assert not fnind
            except Exception:
                mismatches = np.unique(dninf + fnind)
                msg = ("File and description keys do not match. "
                       "Problematic keys: " + ", ".join(mismatches))
                raise AssertionError(msg)
        else:
            desc_dict = {key: None for key in file_dict.keys()}

        # Let's remove existing keys here
        if not overwrite:
            with h5py.File(self.excl_fpath, "r") as h5:
                keys = list(h5.keys())
                for key in keys:
                    if key in file_dict:
                        del file_dict[key]
                        del desc_dict[key]

        # Can we parallelize this?
        for dname, file in tqdm(file_dict.items(), total=len(file_dict)):
            description = desc_dict[dname]
            self.add_layer(dname, file, description, overwrite=overwrite)

    def _check_dims(self, path):
        # Check new layers against the first6 added raster
        if self.profile is not None:
            old = self.profile
            dname = os.path.basename(path)
            with rio.open(path, "r") as r:
                new = r.profile

            # Check the CRS
            if not crs_match(old["crs"], new["crs"]):
                raise AssertionError(f"CRS for {dname} does not match "
                                     "exisitng CRS.")

            # Check the transform
            try:
                # Standardize these
                old_trans = old["transform"][:6]
                new_trans = new["transform"][:6]
                assert [old_trans[i] == new_trans[i] for i in range(6)]
            except AssertionError:
                print(f"Geotransform for {dname} does not match geotransform.")
                raise

            # Check the dimesions
            try:
                assert old["width"] == new["width"]
                assert old["height"] == new["height"]
            except AssertionError:
                print(f"Width and/or height for {dname} does not match "
                      "existing dimensions.")
                raise

    def _convert_coords(self, xs, ys):
        # Convert projected coordinates into WGS84
        print("Transforming xy...")
        # mx, my = np.meshgrid(xs, ys)
        crs = CRS(self.profile["crs"])
        if "World Geodetic System 1984" in crs.datum.name:
            tcrs = CRS("epsg:4326")
        elif "North American Datum 1983" in crs.datum.name:
            tcrs = CRS("epsg:4269")
        elif "Unkown" in crs.datum.name:
            tcrs = CRS("epsg:4269")
        else:
            tcrs = CRS("epsg:4326")

        # transformer = Transformer.from_crs(crs, tcrs, always_xy=True)
        # lons, lats = transformer.transform(mx, my)

        # Added because original takes lots of memory for higher resolution
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)

        corners = [
            geometry.Point(min_x, min_y),
            geometry.Point(min_x, max_y),
            geometry.Point(max_x, min_y),
            geometry.Point(max_x, max_y)
        ]
        gdf = gpd.GeoDataFrame({
            "geometry": corners
        }, geometry=corners, crs=crs)
        gdf = gdf.to_crs(tcrs)

        x_corners, y_corners = zip(*[(point.x, point.y)
                                     for point in gdf.geometry])
        x_trans = np.linspace(np.min(x_corners), np.max(x_corners),
                              num=len(xs), dtype="float32")
        y_trans = np.linspace(np.min(y_corners), np.max(y_corners),
                              num=len(ys), dtype="float32")

        lons, lats = np.meshgrid(x_trans, y_trans)

        return lons, lats

    def _get_coords(self):
        # Get x and y coordinates (One day we'll have one transform order!)
        geom = self.profile["transform"]  # Ensure its in the right order
        xres = geom[0]
        ulx = geom[2]
        yres = geom[4]
        uly = geom[5]

        # Not doing rotations here
        xs = [ulx + col * xres for col in range(self.profile["width"])]
        ys = [uly + row * yres for row in range(self.profile["height"])]

        # Let's not use float 64
        xs = np.array(xs).astype("float32")
        ys = np.array(ys).astype("float32")

        return xs, ys

    def _initialize_h5(self):
        # Create an empty hdf file if one doesn't exist
        date = format(dt.datetime.today(), "%Y-%m-%d %H:%M")
        self.excl_fpath = os.path.expanduser(self.excl_fpath)
        self.excl_fpath = os.path.abspath(self.excl_fpath)
        if not os.path.exists(self.excl_fpath):
            os.makedirs(os.path.dirname(self.excl_fpath), exist_ok=True)
            with h5py.File(self.excl_fpath, "w") as ds:
                ds.attrs["creation_date"] = date

        # Update attributes if provided
        if self.attrs:
            with h5py.File(self.excl_fpath, "r+") as ds:
                for akey, attr in self.attrs.items():
                    if isinstance(attr, (list, dict)):
                        attr = json.dumps(attr)
                    ds.attrs[akey] = attr

    def _preflight(self):
        """More initializing steps."""
        self._set_profile()
        if self.lookup:
            if not isinstance(self.lookup, dict):
                with open(self.lookup, "r") as file:
                    self.lookup = json.load(file)

    def _set_coords(self, profile):
        # Add the lat and lon meshgrids if they aren't already present
        with h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            attrs = hdf.attrs.keys()

            # Set profile if needed
            if "profile" not in attrs:
                hdf.attrs["profile"] = json.dumps(profile)
            self.profile = profile

            # Set coordinates if needed
            if "latitude" not in keys or "longitude" not in keys:
                # Get the original crs coordinates
                xs, ys = self._get_coords()

                # Convert to geographic coordinates
                lons, lats = self._convert_coords(xs, ys)
                print("setting_coords")

                # Create grid and upload
                hdf.create_dataset(name="longitude", data=lons)
                hdf.create_dataset(name="latitude", data=lats)

    def _set_profile(self):
        """Return the exclusion file profile if available."""
        profile = None
        if os.path.exists(self.excl_fpath):
            with h5py.File(self.excl_fpath, "r") as ds:
                if "profile" in ds.attrs.keys():
                    profile = json.loads(ds.attrs["profile"])
        self.profile = profile


class Reformatter(Exclusions):
    """Reformat any file or set of files into a reV-shaped raster."""

    def __init__(self, inputs, out_dir=".", template=None, excl_fpath=None,
                 overwrite_tif=False, overwrite_dset=False, attrs=None,
                 lookup=None, multithread=True, parallel=False,
                 gdal_cache_max=None):
        """Initialize Reformatter object.

        Parameters
        ----------
        inputs : str | dict | pd.core.frame.DataFrame
            Input data information. If dictionary, top-level keys provide
            the target name of reformatted dataset, second-level keys are
            'path' (required path to file'), 'field' (optional field name for
            vectors), 'buffer' (optional buffer distance), and 'layer'
            (required only for FileGeoDatabases). Inputs can also include any
            additional field and it will be included in the attributes of the
            HDF5 dataset if writing to and HDF5 file. If pandas data frame, the
            secondary keys are required as columns, and the top-level key
            is stored in a 'name' column. A path to a CSV of this table is
            also acceptable.
        out_dir : str
            Path to a directory where reformatted data will be written as
            GeoTiffs. Defaults to current directory.
        template : str
            Path to a raster with target georeferencing attributes. If 'None'
            the 'excl_fpath' must point to an HDF5 file with target
            georeferencing information as a top-level attribute.
        excl_fpath : str
            Path to existing or target HDF5 exclusion file. If provided,
            reformatted datasets will be added to this file.
        overwrite_tif : boolean
            If `True`, this will overwrite rasters in `out_dir`.
        overwrite_dset : boolean
            If `True`, this will overwrite datasets in `excl_fpath`.
        attrs : dict
            Dictionary of top-level attributes. Will overwrite existing
            attributes.
        lookup : str | dict
            Dictionary or path dictionary of raster value, key pairs
            derived from shapefiles containing string values (optional).
        multithread : boolean
            Use mutlithreading in each GDAL process (all available threads).
        parallel : boolean
             Process each dataset in parallel (all cores - 1).
        gdal_cache_max : int
            Maximum cache storage in MW. If not set, it uses the GDAL default.
        """
        super().__init__(str(excl_fpath), attrs, lookup)

        os.makedirs(out_dir, exist_ok=True)

        if excl_fpath is not None:
            self.excl_fpath = os.path.abspath(os.path.expanduser(excl_fpath))
        else:
            self.excl_fpath = excl_fpath

        self._parse_inputs(inputs)

        if template is not None:
            self.template = os.path.abspath(os.path.expanduser(template))
        else:
            self.template = template

        self.out_dir = os.path.abspath(os.path.expanduser(out_dir))
        self.overwrite_tif = overwrite_tif
        self.overwrite_dset = overwrite_dset
        self.attrs = attrs
        self.multithread = multithread
        self.parallel = parallel
        self.gdal_cache_max = gdal_cache_max

        if lookup:
            self.lookup = lookup
        else:
            self.lookup = {}

    def __repr__(self):
        """Return object representation string."""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() if k != "inputs"]
        pattrs = ", ".join(attrs)
        return f"<Reformatter: {pattrs}>"

    @property
    def creation_options(self):
        """Return standard raster creation options."""
        ops = {
            "compress": "lzw",
            "tiled": "yes",
            "blockxsize": 128,  # How to optimize internal tiling?
            "blockysize": 128,
            "BIGTIFF": "YES"
        }
        return ops

    @property
    def meta(self):
        """Return the meta information from the template file."""
        meta = None
        if self.template is not None:
            with rio.open(self.template) as raster:
                meta = raster.profile
        else:
            if not os.path.exists(self.excl_fpath):
                raise FileNotFoundError("If not template is provided, an "
                                        "existing `excl_fpath` is required.")
            with h5py.File(self.excl_fpath, "r") as ds:
                if "profile" in ds.attrs:
                    meta = json.loads(ds.attrs["profile"])
                else:
                    for key, data in ds.items():
                        if len(data.shape) == 3:
                            if "profile" in data.attrs:
                                meta = json.loads(data.attrs["profile"])
                                break

        if not meta:
            raise ValueError("No meta object found.")

        return meta

    @property
    def rasters(self):
        """Return list of all rasters in inputs."""
        rasters = {}
        for name, attrs in self.inputs.items():
            if str(attrs["path"]).split(".")[-1] == "tif":
                rasters[name] = attrs

        return rasters

    def reformat_rasters(self):
        """Reformat all raster files in inputs."""
        # Sequential for now
        print(f"Formatting {len(self.rasters)} rasters...")
        if self.parallel:
            dsts = self._reformat_rasters_parallel()
        else:
            dsts = self._reformat_rasters_serial()
        return dsts

    def reformat_raster(self, path, dst):
        """Resample and re-project a raster."""
        # Check that the input path exists
        if not os.path.exists(path):
            raise OSError(f"{path} does not exist.")

        # Run warp if needed
        if os.path.exists(dst) and self.overwrite_tif:
            os.remove(dst)
        warp(
            src=path,
            dst=dst,
            template=self.template,
            creation_ops=self.creation_options,
            multithread=self.multithread,
            cache_max=self.gdal_cache_max
        )

    def reformat_vectors(self):
        """Reformat all vector files in inputs."""
        # Sequential processing for now
        dsts = []
        print(f"Formatting {len(self.vectors)} vectors...")
        for name, attrs in tqdm(self.vectors.items()):

            # Unpack attributes
            path = attrs["path"]
            if "field" in attrs:
                field = attrs["field"]
            else:
                field = None

            if "buffer" in attrs:
                buffer = attrs["buffer"]
            else:
                buffer = None

            # Create destination path
            dst = os.path.join(self.out_dir, f"{name}.tif")
            dsts.append(dst)

            # Reformat vector
            self.reformat_vector(
                name=name,
                path=path,
                dst=dst,
                field=field,
                buffer=buffer
            )

            # Add formatted path to input
            self.inputs[name]["formatted_path"] = dst

        return dsts

    def reformat_vector(self, name, path, dst, field=None,  buffer=None):
        """Preprocess, re-project, and rasterize a vector."""
        # Check that path exists
        if not os.path.exists(path):
            raise OSError(f"{path} does not exist.")

        # Remove if overwrite
        if self.overwrite_tif and os.path.exists(dst):
            os.remove(dst)
        # Read and process file
        gdf = self._process_vector(name, path, field, buffer)
        if not os.path.exists(dst):
            meta = self.meta

            # Rasterize
            elements = gdf.values
            shapes = [(g, r) for r, g in elements]
            out_shape = [meta["height"], meta["width"]]
            transform = meta["transform"]
            with rio.Env():
                array = features.rasterize(shapes, out_shape, all_touched=True,
                                           transform=transform, dtype="uint8")

            # Attempt to set best dtype
            dtype = str(array.dtype)
            if "int" in dtype:
                nodata = np.iinfo(dtype).max
            else:
                nodata = np.finfo(dtype).max
            meta["dtype"] = dtype
            meta["nodata"] = nodata

            # Write to a raster
            with rio.Env():
                with rio.open(dst, "w", **meta) as rio_dst:
                    rio_dst.write(array, 1)

    def to_h5(self):
        """transform all formatted rasters into a h5 file"""
        exclusions = Exclusions(self.excl_fpath, self.attrs, self.lookup)
        for name, attrs in tqdm(self.inputs.items(), total=len(self.inputs)):
            if "formatted_path" in attrs:
                file = attrs["formatted_path"]
                exclusions.add_layer(name, file, overwrite=self.overwrite_dset)

                # Add attributes
                with h5py.File(self.excl_fpath, "r+") as ds:
                    for akey, attr in attrs.items():
                        if attr:
                            if isinstance(attr, (dict, list)):
                                attr = json.dumps(attr)
                            if isinstance(attr, PosixPath):
                                attr = str(attr)
                            ds[name].attrs[akey] = attr

    @property
    def vectors(self):
        """Return list of all shapefiles in inputs."""
        vectors = {}
        for name, attrs in self.inputs.items():
            if not os.path.exists(attrs["path"]):
                raise OSError(f"{attrs['path']} does not exist.")
            if str(attrs["path"]).split(".")[-1] in ["gpkg", "shp", "geojson"]:
                vectors[name] = attrs
        return vectors

    def _check_input_fields(self, inputs):
        """Check that an input table or CSV contains all needed fields."""
        check = all([field in inputs.columns for field in NEEDED_FIELDS])
        assert check, ("Not all required fields present in inputs: "
                       f"{NEEDED_FIELDS}")

    def _parse_inputs(self, inputs):
        """Take input values and create standard dictionary."""
        # Create dictionary
        if isinstance(inputs, dict):
            self.inputs = inputs
        else:
            self._check_input_fields(inputs)
            self.inputs = {}
            if isinstance(inputs, str):
                inputs = pd.read_csv(inputs)
            inputs = inputs[~pd.isnull(inputs["name"])]
            inputs = inputs[~pd.isnull(inputs["path"])]
            inputs = inputs[inputs["name"] != "\xa0"]
            inputs = inputs[inputs["path"] != "\xa0"]

            cols = [c for c in inputs.columns if c != "name"]
            for i, row in inputs.iterrows():
                self.inputs[row["name"]] = {}
                for col in cols:
                    self.inputs[row["name"]][col] = row[col]

        # Convert NaNs to None
        for key, attrs in self.inputs.items():
            for akey, attr in attrs.items():
                if not isinstance(attr, (str, PosixPath, dict)):
                    if attr is not None:
                        if np.isnan(attr):
                            self.inputs[key][akey] = None

    def _map_strings(self, layer_name, gdf, field):
        """Map string values to integers and save a lookup dictionary."""
        # Assing integers to unique string values
        strings = gdf["raster_value"].unique()
        string_map = {i + 1: v for i, v in enumerate(strings)}
        value_map = {v: k for k, v in string_map.items()}

        # Replace strings with integers
        gdf["raster_value"] = gdf[field].map(value_map)

        # Update the string lookup dictionary
        self.lookup[layer_name] = string_map

        return gdf

    def _process_vector(self, name, path, field=None, buffer=None):
        """Process a single vector file."""
        # Read in file and check the path
        gdf = gpd.read_file(path)

        if gdf.shape[0] == 0:
            raise IndexError(f"{path} is an empty file.")

        # Check the projection;
        if not crs_match(gdf.crs, self.meta["crs"]):
            gdf = gdf.to_crs(self.meta["crs"])

        # Check if the field value in the shapefile and Assign raster value
        if field:
            if field not in gdf.columns:
                raise Exception(f"Field '{field}' not in '{path}'")
            gdf["raster_value"] = gdf[field]
        else:
            gdf["raster_value"] = 1

        # Account for string values
        sample = gdf["raster_value"].iloc[0]
        if isinstance(sample, str):
            if isint(sample):
                gdf["raster_value"] = gdf["raster_value"].astype(int)
            elif isfloat(sample):
                gdf["raster_value"] = gdf["raster_value"].astype(float)
            else:
                gdf = self._map_strings(name, gdf, field)

        # Reduce to two fields
        gdf = gdf[["raster_value", "geometry"]]

        # Buffering
        if buffer:
            gdf["geometry"] = gdf["geometry"].buffer(buffer)

        return gdf

    def _reformat_rasters_parallel(self):
        # Collect arguments and destination paths
        args = []
        dsts = []
        for name, attrs in self.rasters.items():
            path = str(attrs["path"])
            dst = os.path.join(self.out_dir, f"{name}.tif")
            self.inputs[name]["formatted_path"] = dst
            dsts.append(dst)
            args.append((path, dst))

        # Run the reformatting method
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.starmap(self.reformat_raster, args):
                pass

        return dsts

    def _reformat_rasters_serial(self):
        # Run the reformatting method and collect destination paths
        dsts = []
        for name, attrs in self.rasters.items():
            path = str(attrs["path"])
            dst = os.path.join(self.out_dir, f"{name}.tif")
            dsts.append(dst)
            self.reformat_raster(path=path, dst=dst)
            self.inputs[name]["formatted_path"] = dst
        return dsts

    def main(self):
        """Reformat all vectors and rasters listed in the inputs."""
        _ = self.reformat_rasters()
        _ = self.reformat_vectors()
        if self.excl_fpath:
            print(f"Building/updating exclusion {self.excl_fpath}...")
            self.to_h5()
