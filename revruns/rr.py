"""Revruns Functions.

Almost all of the functionality is currently stored in the CLI scripts to
avoid the load time needed to load shared functions, but anything in here can
be accessed directly from revruns. Place new functions that might be useful
in the future here.

Created on Wed Dec  4 07:58:42 2019

@author: twillia2
"""
import json
import multiprocessing as mp
import os
import warnings
        

from glob import glob
from json import JSONDecodeError

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd

from pathos.multiprocessing import ProcessingPool as Pool
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point
from tqdm import tqdm


pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 20)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def crs_match(crs1, crs2):
    """Check if two coordinate reference systems match."""
    # Using strings and CRS objects directly is not consistent enough
    check = True
    crs1 = CRS(crs1).to_dict()
    crs2 = CRS(crs2).to_dict()
    for key, value in crs1.items():
        if key in crs2:
            if value != crs2[key]:
                return False
    return check


def crs_match_alt(crs1, crs2):
    """Alternative CRS match with no extra dependencies."""
    # Check that this a proj4 string
    # it would have +lat_1 +lat_2 etc
    assert "+lat_1" in crs1, "Fail"
    assert "+lat_1" in crs2, "Fail"

    # Check that there is no difference
    diff = set(crs1.split())  - set(crs2.split())

    if not diff:
        return True
    else:
        return False


def get_sheet(file_name, sheet_name=None, starty=0, startx=0, header=0):
    """Read in/check available sheets from an excel spreadsheet file."""
    from xlrd import XLRDError

    # Open file
    file = pd.ExcelFile(file_name)
    sheets = file.sheet_names

    # Run with no sheet_name for a list of available sheets
    if not sheet_name:
        print("No sheet specified, returning a list of available sheets.")
        return sheets
    if sheet_name not in sheets:
        raise ValueError(sheet_name + " not in file.")

    # Try to open sheet, print options if it fails
    try:
        table = file.parse(sheet_name=sheet_name, header=header)
    except XLRDError:
        print(sheet_name + " is not available. Available sheets:\n")
        for s in sheets:
            print("   " + s)

    return table


def h5_to_csv(src, dst, dataset):
    """Reformat a reV outpur HDF5 file/dataset to a csv."""
    # Read in meta, time index, and data
    with h5py.File(src, "r") as ds:
        # Get an good time index
        if "multi" in src:
            time_key = [ti for ti in ds.keys() if "time_index" in ti][0]
            data = [ds[d][:] for d in ds.keys() if dataset in d]
            data = np.array(data)
        else:
            time_key = "time_index"
            data = ds[dataset][:]

    # Read in needed elements
    meta = pd.DataFrame(ds["meta"][:])
    time_index = [t.decode()[:-9] for t in ds[time_key][:]]

    # Decode meta, use as base
    meta.rr.decode()

    # If its 1-D just give just append the data to meta
    if len(data.shape) == 1:
        meta[dataset] = data
        df = meta.copy()

    # If its more than 1-D, label the array and use that as the table
    elif len(data.shape) > 1:
        # If its 3-D, its a mult-year, find the mean profile first
        if len(data.shape) == 3:
            time_index = [t[5:] for t in time_index]
            data = np.mean(data, axis=0)

        # Now its def 2-D
        df = pd.DataFrame(data)
        df["time_index"] = time_index
        cols = [str(c) for c in df.columns]
        df.columns = cols
        cols = ["time_index"] + cols[:-1]
        df = df[cols]

    df.to_csv(dst, index=False)

    return 0


def isint(x):
    """Check if character string is an integer."""
    check = False
    if isinstance(x, int):
        check = True
    elif isinstance(x, str):
        if "." not in x:
            try:
                int(x)
                check = True
            except ValueError:
                check = False
    return check


def isfloat(x):
    """Check if character string is an float."""
    try:
        float(x)
        return True
    except ValueError:
        return False


def mode(x):
    """Return the mode of a list of values."""  # <---------------------------- Works with numpy's max might break here
    return max(set(x), key=x.count)


def par_apply(df, field, fun):
    """Apply a function in parallel to a pandas data frame field."""
    import numpy as np
    import pathos.multiprocessing as mp

    from tqdm import tqdm

    def single_apply(arg):
        """Apply a function to a pandas data frame field."""
        cdf, field, fun = arg
        try:
            values = cdf[field].apply(fun)
        except Exception:
            raise
        return values

    ncpu = mp.cpu_count()
    cdfs = np.array_split(df, ncpu)
    args = [(cdf, field, fun) for cdf in cdfs]

    values = []
    with mp.Pool(ncpu) as pool:
        for value in tqdm(pool.imap(single_apply, args), total=ncpu):
            values.append(value)
    values = [v for sv in values for v in sv]

    return values


def write_config(config_dict, path):
    """Write a configuration dictionary to a json file."""
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))


class Data_Path:
    """Data_Path joins a root directory path to data file paths."""

    def __init__(self, data_path=".", mkdir=False, warnings=True):
        """Initialize Data_Path."""
        data_path = os.path.abspath(os.path.expanduser(data_path))
        self.data_path = data_path
        self.last_path = os.getcwd()
        self.warnings = warnings
        self._exist_check(data_path, mkdir)
        self._expand_check()

    def __repr__(self):
        """Print the data path."""
        items = ["=".join([str(k), str(v)]) for k, v in self.__dict__.items()]
        arguments = ", ".join(items)
        msg = "".join(["<Data_Path " + arguments + ">"])
        return msg

    def contents(self, *args, recursive=False):
        """List all content in the data_path or in sub directories."""
        if not any(["*" in a for a in args]):
            items = glob(self.join(*args, "*"), recursive=recursive)
        else:
            items = glob(self.join(*args), recursive=recursive)
        items.sort()
        return items

    def extend(self, path, mkdir=False):
        """Return a new Data_Path object with an extended home directory."""
        new = Data_Path(os.path.join(self.data_path, path), mkdir)
        return new

    def folders(self, *args, recursive=False):
        """List folders in the data_path or in sub directories."""
        items = self.contents(*args, recursive=recursive)
        folders = [i for i in items if os.path.isdir(i)]
        return folders

    def files(self, *args, recursive=False):
        """List files in the data_path or in sub directories."""
        items = self.contents(*args, recursive=recursive)
        files = [i for i in items if os.path.isfile(i)]
        return files

    def join(self, *args, mkdir=False):
        """Join a file path to the root directory path."""
        path = os.path.join(self.data_path, *args)
        self._exist_check(path, mkdir)
        path = os.path.abspath(path)
        return path

    @property
    def base(self):
        """Return the base name of the home directory."""
        return os.path.basename(self.data_path)

    @property
    def back(self):
        """Change directory back to last working directory if home was used."""
        os.chdir(self.last_path)
        print(self.last_path)

    @property
    def home(self):
        """Change directories to the data path."""
        self.last_path = os.getcwd()
        os.chdir(self.data_path)
        print(self.data_path)


    def _exist_check(self, path, mkdir=False):
        """Check if the directory of a path exists, and make it if not."""
        # If this is a file name, get the directory
        if "." in path:  # Will break if you use "."'s in your directories
            directory = os.path.dirname(path)
        else:
            directory = path

        # Don't try this with glob patterns
        if "*" not in directory:
            if not os.path.exists(directory):
                if mkdir:
                    if self.warnings:
                        print(f"Warning: {directory} did not exist, "
                              "creating directory.")
                    os.makedirs(directory, exist_ok=True)
                else:
                    if self.warnings:
                        print(f"Warning: {directory} does not exist.")

    def _expand_check(self):
        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)


@pd.api.extensions.register_dataframe_accessor("rr")
class PandasExtension:
    """Accessing useful pandas functions directly from a data frame object."""

    def __init__(self, pandas_obj):
        """Initialize PandasExtension object."""
        warnings.simplefilter(action='ignore', category=UserWarning)
        if type(pandas_obj) != pd.core.frame.DataFrame:
            if type(pandas_obj) != gpd.geodataframe.GeoDataFrame:
                raise TypeError("Can only use .rr accessor with a pandas or "
                                "geopandas data frame.")
        self._obj = pandas_obj

    def average(self, value, weight="n_gids", group=None):
        """Return the weighted average of a column.

        Parameters
        ----------
        value : str
            Column name of the variable to calculate.
        weight : str
            Column name of the variable to use as the weights. The default is
            'n_gids'.
        group : str, optional
            Column name of the variable to use to group results. The default is
            None.

        Returns
        -------
        dict | float
            Single value or a dictionary with group, weighted average value
            pairs.
        """
        df = self._obj.copy()
        if not group:
            values = df[value].values
            weights = df[weight].values
            x = np.average(values, weights=weights)
        else:
            x = {}
            for g in df[group].unique():
                gdf = df[df[group] == g]
                values = gdf[value].values
                weights = gdf[weight].values
                x[g] = np.average(values, weights=weights)
        return x

    def bmap(self):
        """Show a map of the data frame with a basemap if possible."""
        if not isinstance(self._obj, gpd.geodataframe.GeoDataFrame):
            print("Data frame is not a GeoDataFrame")

    def decode(self):
        """Decode the columns of a meta data object from a reV output."""
        import ast

        def decode_single(x):
            """Try to decode a single value, pass if fail."""
            try:
                x = x.decode()
            except UnicodeDecodeError:
                x = "indecipherable"
            return x

        for c in self._obj.columns:
            x = self._obj[c].iloc[0]
            if isinstance(x, bytes):
                try:
                    self._obj[c] = self._obj[c].apply(decode_single)
                except Exception:
                    self._obj[c] = None
                    print("Column " + c + " could not be decoded.")
            elif isinstance(x, str):
                try:
                    if isinstance(ast.literal_eval(x), bytes):
                        try:
                            self._obj[c] = self._obj[c].apply(
                                lambda x: ast.literal_eval(x).decode()
                            )
                        except Exception:
                            self._obj[c] = None
                            print("Column " + c + " could not be decoded.")
                except:
                    pass

    def dist_apply(self, linedf):
        """To apply the distance function in parallel (not ready)."""
        # Get distances
        ncpu = os.cpu_count()
        chunks = np.array_split(self._obj.index, ncpu)
        args = [(self._obj.loc[idx], linedf) for idx in chunks]
        distances = []
        with Pool(ncpu) as pool:
            for dists in tqdm(pool.imap(self.point_line, args),
                              total=len(args)):
                distances.append(dists)
        return distances

    def find_coords(self, df=None):
        """Check if lat/lon names are in a pre-made list of possible names."""
        # List all column names
        if df is None:
            df = self._obj.copy()
        cols = df.columns

        # For direct matches
        ynames = ["y", "lat", "latitude", "Latitude", "ylat"]
        xnames = ["x", "lon", "long", "longitude", "Longitude", "xlon",
                  "xlong"]

        # Direct matches
        possible_ys = [c for c in cols if c in ynames]
        possible_xs = [c for c in cols if c in xnames]

        # If no matches return item and rely on manual entry
        if len(possible_ys) == 0 or len(possible_xs) == 0:
            raise ValueError("No field names found for coordinates, use "
                             "latcol and loncol arguments.")

        # If more than one match raise error
        elif len(possible_ys) > 1:
            raise ValueError("Multiple possible entries found for y/latitude "
                             "coordinates, use latcol argument: " +
                             ", ".join(possible_ys))
        elif len(possible_xs) > 1:
            raise ValueError("Multiple possible entries found for y/latitude "
                             "coordinates, use latcol argument: " +
                             ", ".join(possible_xs))

        return possible_ys[0], possible_xs[0]

    def gid_join(self, df_path, fields, agg="mode", left_on="res_gids",
                 right_on="gid"):
        """Join a resource-scale data frame to a supply curve data frame.

        Parameters
        ----------
        df_path : str
            Path to csv with desired join fields.
        fields : str | list
            The field(s) in the right DataFrame to join to the left.
        agg : str
            The aggregating function to apply to the right DataFrame. Any
            appropriate numpy function.
        left_on : str
            Column name to join on in the left DataFrame.
        right_on : str
            Column name to join on in the right DataFrame.

        Returns
        -------
        pandas.core.frame.DataFrame
            A pandas DataFrame with the specified fields in the right
            DataFrame aggregated and joined.
        """
        # The function to apply to each item of the left dataframe field
        def single_join(x, vdict, right_on, field, agg):
            """Return the aggregation of a list of values in df."""
            x = self._destring(x)
            rvalues = [vdict[v] for v in x]
            rvalues = [self._destring(v) for v in rvalues]
            rvalues = [self._delist(v) for v in rvalues]
            return agg(rvalues)

        def chunk_join(arg):
            """Apply single to a subset of the main dataframe."""
            chunk, df_path, left_on, right_on, field, agg = arg
            rdf = pd.read_csv(df_path)
            vdict = dict(zip(rdf[right_on], rdf[field]))
            chunk[field] = chunk[left_on].apply(single_join, args=(
                vdict, right_on, field, agg
                )
            )
            return chunk

        # Create a copy of the left data frame
        df1 = self._obj.copy()

        # Set the function
        if agg == "mode":
            def mode(x): max(set(x), key=x.count)
            agg = mode
        else:
            agg = getattr(np, agg)

        # If a single string is given for the field, make it a list
        if isinstance(fields, str):
            fields = [fields]

        # Split this up and apply the join functions
        chunks = np.array_split(df1, os.cpu_count())
        for field in fields:
            args = [(c, df_path, left_on, right_on, field, agg)
                    for c in chunks]
            df1s = []
            with Pool(os.cpu_count()) as pool:
                for cdf1 in pool.imap(chunk_join, args):
                    df1s.append(cdf1)
            df = pd.concat(df1s)

        return df

    def nearest(self, df2, fields=None, lat=None, lon=None, no_repeat=False,
                k=5):
        """Find all of the closest points in a second data frame.

        Parameters
        ----------
        df2 : pandas.core.frame.DataFrame | geopandas.geodataframe.GeoDataFrame
            The second data frame from which a subset will be extracted to
            match all points in the first data frame.
        fields : str | list
            The field(s) in the second data frame to append to the first.
        lat : str
            The name of the latitude field.
        lon : str
            The name of the longitude field.
        no_repeat : logical
            Return closest points with no duplicates. For two points in the
            left dataframe that would join to the same point in the right, the
            point of the left pair that is closest will be associated with the
            original point in the right, and other will be associated with the
            next closest. (not implemented yet)
        k : int
            The number of next closest points to calculate when no_repeat is
            set to True. If no_repeat is false, this value is 1.

        Returns
        -------
        df : pandas.core.frame.DataFrame | geopandas.geodataframe.GeoDataFrame
            A copy of the first data frame with the specified field and a
            distance column.
        """
        # Pull out the target data frame
        df1 = self._obj.copy()

        # Locate the coordinate fields for the target data frame
        x1 = "x"
        y1 = "y"
        if isinstance(df1, gpd.geodataframe.GeoDataFrame):
            df1["x"] = df1["geometry"].x
            df1["y"] = df1["geometry"].y
        else:
            y1, x1 = self.find_coords(df1)

        # Locate the coordinate fields for the second data frame
        x2 = "x"
        y2 = "y"
        if isinstance(df1, gpd.geodataframe.GeoDataFrame):
            df2["x"] = df2["geometry"].x
            df2["y"] = df2["geometry"].y
        else:
            y2, x2 = self.find_coords(df2)

        # Choose target fields from second data frame
        if fields:
            if isinstance(fields, str):
                fields = [fields]
        else:
            fields = [c for c in df if c not in [x2, y2, "geometry"]]

        # Get arrays of point coordinates
        crds1 = df1[[x1, y1]].values
        crds2 = df2[[x2, y2]].values

        # Build the connections tree and query points from the first df
        tree = cKDTree(crds2)
        if no_repeat:
            dist, idx = tree.query(crds1, k=k)
            dist, idx = self._derepeat(dist, idx)
        else:
            dist, idx = tree.query(crds1, k=1)

        # We might be relacing a column
        for field in fields:
            if field in df1:
                del df1[field]

        # Rebuild the dataset
        dfa = df1.reset_index(drop=True)
        dfb = df2.iloc[idx, :]
        dfb = dfb.reset_index(drop=True)
        df = pd.concat([dfa, dfb[fields], pd.Series(dist, name='dist')],
                       axis=1)

        return df

    # def papply(self, func, **kwargs):
    #     """Apply a function to a dataframe in parallel chunks."""
    #     from pathos import multiprocessing as mp

    #     from itertools import product

    #     df = self._obj.copy()
    #     cdfs = np.array_split(df, os.cpu_count() - 1)
    #     pool = mp.Pool(mp.cpu_count() - 1)
    #     args = [(cdf, kwargs) for cdf in cdfs]
    #     out = pool.starmap(cfunc, args)

    def scatter(self, x="capacity", y="mean_lcoe", z=None, color="mean_lcoe",
                size=None):
        """Create a plotly scatterplot."""
        import plotly.express as px

        df = self._obj.copy()

        if z is None:
            fig = px.scatter(df, x, y, color=color, size=size)
        else:
            fig = px.scatter_3d(df, x, y, z, color=color, size=size)

        fig.show()

    def to_bbox(self, bbox):
        """Return points within a bounding box [xmin, ymin, xmax, ymax]."""
        df = self._obj.copy()
        df = df[(df["longitude"] >= bbox[0]) &
                (df["latitude"] >= bbox[1]) &
                (df["longitude"] <= bbox[2]) &
                (df["latitude"] <= bbox[3])]
        return df

    def to_geo(self, lat=None, lon=None):
        """Convert a Pandas data frame to a geopandas geodata frame."""
        # Let's not transform in place
        df = self._obj.copy()
        df.rr.decode()

        # Find coordinate columns
        if not isinstance(df, gpd.geodataframe.GeoDataFrame):
            if "geometry" not in df.columns:
                if "geom" not in df.columns:
                    try:
                        lat, lon = self.find_coords()
                    except ValueError:
                        pass

                    # For a single row
                    def to_point(x):
                        return Point(x.values)
                    df["geometry"] = df[[lon, lat]].apply(to_point, axis=1)

                # Create the geodataframe - add in projections
                if "geometry" in df.columns:
                    gdf = gpd.GeoDataFrame(df, crs='epsg:4326',
                                                geometry="geometry")
                if "geom" in df.columns:
                    gdf = gpd.GeoDataFrame(df, crs='epsg:4326',
                                                geometry="geom")
            else:
                gdf = df

        return gdf

    def to_sarray(self):
        """Create a structured array for storing in HDF5 files."""
        # Create a copy
        df = self._obj.copy()

        # For a single column
        def make_col_type(col, types):

            coltype = types[col]
            column = df.loc[:, col]

            try:
                if 'numpy.object_' in str(coltype.type):
                    maxlens = column.dropna().str.len()
                    if maxlens.any():
                        maxlen = maxlens.max().astype(int)
                        coltype = ('S%s' % maxlen)
                    else:
                        coltype = 'f2'
                return column.name, coltype
            except:
                print(column.name, coltype, coltype.type, type(column))
                raise

        # All values and types
        v = df.values
        types = df.dtypes
        struct_types = [make_col_type(col, types) for col in df.columns]
        dtypes = np.dtype(struct_types)

        # The target empty array
        array = np.zeros(v.shape[0], dtypes)

        # For each type fill in the empty array
        for (i, k) in enumerate(array.dtype.names):
            try:
                sample = df[k].iloc[0]
                if dtypes[i].str.startswith('|S') and isinstance(sample, str):
                    array[k] = df[k].str.encode('utf-8').astype('S')
                else:
                    array[k] = v[:, i]
            except:
                raise

        return array, dtypes

    def _delist(self, value):
        """Extract the value of an object if it is a list with one value."""
        if isinstance(value, list):
            if len(value) == 1:
                value = value[0]
        return value

    # def _derepeat(dist, idx):
    #     """Find the next closest index for repeating cKDTree outputs."""  # <-- Rethink this, we could autmomatically set k to the max repeats
    #     # Get repeated idx and counts
    #     k = idx.shape[1]
    #     uiidx, nrepeats = np.unique(idx, return_counts=True)
    #     max_repeats = nrepeats.max()
    #     if max_repeats > k:
    #         raise ValueError("There are a maximum of " + str(max_repeats) +
    #                          " repeating points, to use the next closest "
    #                          "neighbors to avoid repeats, set k to this "
    #                          "number.")

    #     for i in range(k - 1):
    #         # New arrays for this axis
    #         iidx = idx[:, i]
    #         idist = dist[:, i]

    def _destring(self, string):
        """Destring values into their literal python types if needed."""
        try:
            return json.loads(string)
        except (TypeError, JSONDecodeError):
            return string


class Profiles:
    """Methods for manipulating generation profiles."""

    def __init__(self, gen_fpath):
        """Initialize Profiles object.

        Parameters
        ----------
        gen_fpath : str
            Path to a reV generation or representative profile file.
        """
        self.gen_fpath = gen_fpath

    def __repr__(self):
        """Return representation string for Profiles object."""
        return f"<Profiles: gen_fpath={self.gen_fpath}"

    def _best(self, row, gen_fpath, variable, lowest):
        """Find the best generation point in a supply curve table row.

        Parameters
        ----------
        row : pd.core.series.Series
            A row from a reV suuply curve table.
        ds : h5py._hl.files.File
            An open h5py file

        Returns
        -------
        int : The index position of the best profile.
        """
        idx = json.loads(row["gen_gids"])  # Don't forget to add res_gid
        idx.sort()
        with h5py.File(gen_fpath) as ds:
            if lowest:
                gid = idx[np.argmin(ds[variable][idx])]
            else:
                gid = idx[np.argmax(ds[variable][idx])]
            value = ds[variable][gid]
        row["best_gen_gid"] = gid
        row[self._best_name(variable, lowest)] = value
        return row

    def _all(self, idx, gen_fpath, variable):
        """Find the best generation point in a supply curve table row.

        Parameters
        ----------
        row : pd.core.series.Series
            A row from a reV suuply curve table.
        ds : h5py._hl.files.File
            An open h5py file


        Returns
        -------
        int
            The index position of the best profile.
        """
        idx = json.loads(idx)
        idx.sort()
        with h5py.File(gen_fpath) as ds:
            values = ds[variable][idx]
        return values

    def _best_name(self, variable, lowest):
        """Return the column name of the best value column."""
        if lowest:
            name = f"{variable}_min"
        else:
            name = f"{variable}_max"
        return name

    def _derepeat(self, df):
        master_gids = []
        df["gen_gids"] = df["gen_gids"].apply(json.loads)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            gids = row["gen_gids"]
            for g in gids:
                if g in master_gids:
                    gids.remove(g)
                else:
                    master_gids.append(g)

    def get_table(self, sc_fpath, variable, lowest):
        """Apply the _best function in parallel."""
        from pathos import multiprocessing as mp

        # Function to apply to each chunk defined below
        def cfunc(args):
            cdf, gen_fpath, variable, lowest = args
            out = cdf.apply(self._best, gen_fpath=gen_fpath, variable=variable,
                            lowest=lowest, axis=1)
            return out

        # The supply curve data frame
        df = pd.read_csv(sc_fpath)

        # Split the data frame up into chuncks
        ncpu = mp.cpu_count() - 1
        cdfs = np.array_split(df, ncpu)
        arg_list = [(cdf, self.gen_fpath, variable, lowest) for cdf in cdfs]

        # Apply the chunk function in parallel
        outs = []
        with mp.Pool(ncpu) as pool:
            for out in pool.imap(cfunc, arg_list):
                outs.append(out)

        # Concat the outputs back into a single dataframe and sort
        df = pd.concat(outs)
        df = df.sort_values("best_gen_gid")

        return df

    def main(self, sc_fpath, dst, variable="lcoe_fcr-means", lowest=True):
        """Write a dataset of the 'best' profiles within each sc point.

        Parameters
        ----------
        sc_fpath : str
            Path to a supply-curve output table.
        dst : str
            Path to the output HDF5 file.
        variable : str
            Variable to use as selection criteria.
        lowest : boolean
            Select based on the lowest criteria value.
        """
        # Read in the expanded gen value table
        df = pd.read_csv(sc_fpath)
        df["values"] = df["gen_gids"].apply(self._all,
                                            gen_fpath=self.gen_fpath,
                                            variable=variable)

        # Choose which gids to keep to avoid overlap
        tdf = df[["sc_point_gid", "gen_gids", "values"]]

        # Read in the preset table
        df = self.get_table(sc_fpath, variable=variable, lowest=lowest)

        # Subset for common set of fields
        # ...

        # Convert the supply curve table a structure, will use as meta
        sdf, dtypes = df.rr.to_sarray()

        # Create new dataset
        ods = h5py.File(self.gen_fpath, "r")
        nds = h5py.File(dst, "w")

        # One or multiple years?
        keys = [key for key in ods.keys() if "cf_profile" in key]

        # Build the array
        arrays = []
        gids = df["best_gen_gid"].values
        for key in keys:
            gen_array = ods[key][:, gids]


class Financing:
    """Methods for calculating various financing figures in the 2022 ATB."""

    def __init__(self, ir=0.015, i=0.025, rroe=.052, df=0.735, tr=.257):
        """Initialize Financing object.

        Parameters
        ----------
        ir : float
            Interest rate.
        i : float
            Infalation rate.
        rroe : float
            Return on equity.
        df : float
            Debt fraction.
        tr : float
            Tax rate.
        """
        super().__init__()
        self.ir = ir
        self.i = i
        self.rroe = rroe
        self.df = df
        self.tr = tr

    def fcr(self, lifetime=30):
        """Calculate FCR with standard ATB assumptions but variable lifetime."""
        # Cacluate weight avg cost of capital and present value of depreciation
        wacc = self.wacc()
        pvd = self.pvd()

        # Calculate the cost recovery factor
        crf = wacc * (1 / (1 - (1 / (1 + wacc) ** lifetime)))

        # Project finance factor
        profinfactor = (1 - self.tr * pvd) / (1 - self.tr)

        # Fixed charge rate
        fcr = crf * profinfactor

        return fcr

    def pvd(self):
        """Return present value of depreciation."""
        # Modified accelerated cost recovery system
        macrs = np.array([.20, .32, .192, .1152, .1152, .0576])

        # Depreciation factors
        df = np.array(
            [ 
                0.9592,
                0.9201,
                0.8826,
                0.8466,
                0.8121,
                0.7790
            ]
        )

        pvd = sum(macrs * df)

        return pvd

    def wacc(self):
        """Calculate weight average cost of capital"""
        term1 = 1 + ((1 - self.df) * ((1 + self.rroe) * (1 + self.i) - 1))
        term2 = (self.df * ((1 + self.ir) * (1 + self.i) - 1) * (1 - self.tr))
        wacc = ((term1 + term2) / (1 + self.i)) - 1
        return wacc


if __name__ == "__main__":
    # Attempting to match ATB 2022
    self = Financing()
    lifetime = 25
    print(self.fcr(lifetime))