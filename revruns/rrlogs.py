"""Check with reV status, output, and error logs.

TODO:
    - See if we can't use pyslurm to speed up the squeue call

    - This was made in a rush before I was fast enough to build properly. Also,
      This is probably the most frequently used revrun cli, so this should
      definitely be a top candidate for a major refactor.
"""
import copy
import datetime as dt
import json
import os
import warnings

from pandarallel import pandarallel

from glob import glob
from pathlib import Path

import click
import h5py
import pandas as pd
import pathos.multiprocessing as mp

from colorama import Fore, Style
from tabulate import tabulate

try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    from pandas.core.common import SettingWithCopyWarning

pandarallel.initialize(progress_bar=False, verbose=0)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
MODULE_HELP = ("The reV module logs to check. Defaults to all modules: gen, "
               "collect, multi-year, aggregation, supply-curve, or "
               "rep-profiles. (str)")
ERROR_HELP = ("A job index. This will print the error log of a specific job. "
              "(int)")
OUT_HELP = ("A job index. This will print the standard output log of a "
            "specific job.(int)")
STATUS_HELP = ("Print jobs with a given status. Option include 'failed' "
               "(or 'f'), 'success' (or 's'), 'pending' (or 'p'), 'running' "
               "(or 'r'), 'submitted' (or 'sb') and 'unsubmitted (or 'usb'). "
               "(str)")
WALK_HELP = ("Walk the given directory structure and return the status of "
             "all jobs found. (boolean")
FULL_PRINT_HELP = ("When printing log outputs (using -o <pid> or -e <pid>) "
                   "print the full content of the file to the terminal. This "
                   "may fill up your entire shell so the default is to limit "
                   "this output to the first 20 lines of of the target log "
                   "file. (boolean)")
SAVE_HELP = ("Write the outputs of an rrlogs call to a csv. (boolean)")
STATS_HELP = ("Print summary statistics instead of status information. Only "
              "works for existing files (i.e., completed files not in "
              "chunk files). (boolean)")
FIELD_HELP = ("Field in dataset to use if request stat summary (defaults to"
              "mean_cf).")
AU_HELP = ("Count AUs used for requested runs. (boolean)")
VERBOSE_HELP = ("Print status data to console. (boolean)")

CONFIG_DICT = {
    "gen": "config_gen.json",
    "bespoke": "config_bespoke.json",
    "collect": "config_collect.json",
    "econ": "config_econ.json",
    "offshore": "config_offshore.json",
    "multi-year": "config_multi-year.json",
    "aggregation": "config_aggregation.son",
    "supply-curve": "config_supply-curve.json",
    "rep-profiles": "config_rep-profiles.json",
    "nrwal": "config_nrwal.json",
    "qaqc": "config_qaqc.json"
}
MODULE_NAMES = {
    "gen": "generation",
    "bespoke": "bespoke",
    "collect": "collect",
    "econ": "econ",
    "offshore": "offshore",
    "multi-year": "multi-year",
    "aggregation": "supply-curve-aggregation",
    "supply-curve": "supply-curve",
    "rep-profiles": "rep-profiles",
    "nrwal": "nrwal",
    "qaqc": "qa-qc",
    "add-reeds-cols": "add-reeds-cols"
}

FAILURE_STRINGS = ["failure", "fail", "failed", "f"]
PENDING_STRINGS = ["pending", "pend", "p"]
RUNNING_STRINGS = ["running", "run", "r"]
SUBMITTED_STRINGS = ["submitted", "submit", "sb"]
SUCCESS_STRINGS = ["successful", "success", "s"]
UNSUBMITTED_STRINGS = ["unsubmitted", "unsubmit", "u"]
PRINT_COLUMNS = ["index", "job_name", "job_status", "pipeline_index",
                 "job_id", "runtime", "date", "date_submitted",
                 "date_completed"]
STAT_COLORS = {
    "count": "\033[30m",
    "mean": "\033[32m",
    "std": "\033[96m",
    "min": "\033[34m",
    "25%": "\033[94m",
    "50%": "\033[93m",
    "75%": "\033[31m",
    "max": "\033[91m"
}


def safe_round(x, n):
    """Round a number to n places if x is a number."""
    try:
        xn = round(x, n)
    except TypeError:
        xn = x
    return xn


def read_h5(fpath, field):
    """Read an h5 file and return dataframe."""
    # Make sure field is in file
    with h5py.File(fpath) as ds:
        fields = list(ds.keys())
    if field not in fields:
        raise KeyError(f"{field} not in {fpath}")

    # Pull data from file
    with h5py.File(fpath) as ds:
        data = ds[field][:]
        if len(data.shape) > 1:
            data = data.mean(axis=0)
        meta = pd.DataFrame(ds["meta"][:])
        meta["gid"] = meta.index
        meta[field] = data
    meta = meta[["gid", field]]
    return meta


class Log_Finder:
    """Methods for finding logging information in a run directory."""


class No_Pipeline(Log_Finder):
    """Methods for checking logs without a reV pipeline."""

    def __init__(self, folder=".", error=None, out=None, walk=False):
        """Initialize NPipeline object."""
        self.folder = folder
        self.error = error
        self.out = out
        self.walk = walk

    @property
    def error_logs(self):
        """Return paths to all error logs."""


class RRLogs(No_Pipeline):
    """Methods for checking rev run statuses."""

    def __init__(self, folder=".", module=None, status=None, error=None,
                 out=None, walk=False, full_print=False, csv=False,
                 stats=False, field="mean_cf", count_aus=False,
                 verbose=True):
        """Initialize an RRLogs object."""
        self.folder = os.path.expanduser(os.path.abspath(folder))
        self.module = module
        self.status = status
        self.error = error
        self.out = out
        self.walk = walk
        self.full_print = full_print
        self.csv = csv
        self.stats = stats
        self.field = field
        self.count_aus = count_aus
        self.verbose = verbose

    def __repr__(self):
        """Return RRLogs object representation string."""
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"<RRLogs object: {attrs}>"

    def build_dataframe(self, sub_folder, module=None):
        """Convert a status entry into dataframe."""
        # Get the status dictionary
        _, status = self.find_status(sub_folder)

        # If the status is actively being updated skip this
        if status == "updating":
            return status

        # There might be a log file with no status data frame
        elif isinstance(status, dict):
            # If just one module
            if module:
                df = self.build_module_dataframe(status, module)
                if df is None:
                    msg = f"{module} not found in status file.\n"
                    print(Fore.RED + msg + Style.RESET_ALL)
                    return

            # Else, create a single data frame with everything
            else:
                modules = status.keys()
                names_modules = {v: k for k, v in MODULE_NAMES.items()}
                dfs = []
                for module_name in modules:
                    module = names_modules[module_name]
                    try:
                        mstatus = self.build_module_dataframe(status, module)
                    except KeyError:
                        print(f"{module} not found in status file.")
                        raise
                    if mstatus is not None:
                        dfs.append(mstatus)
                if not dfs:
                    return
                df = pd.concat(dfs, sort=False)

            # Here, let's improve the time estimation somehow
            if "runtime" not in df.columns and not self.stats:
                df["runtime"] = "nan"

            # And refine this down for the printout
            if not self.stats:
                df["job_name"] = df.index
                df = df.reset_index()
                df["index"] = df.index
                if "job_id" in df:
                    df = self.check_index(df, sub_folder)
                cols = [col for col in PRINT_COLUMNS if col in df]
                df = df[cols]
                df["runtime"] = df["runtime"].apply(safe_round, n=2)

            return df
        else:
            return None

    def build_module_dataframe(self, status, module="gen"):
        """Convert the status entry for a module to a dataframe."""
        # Get the module entry
        cstatus = copy.deepcopy(status)
        mkey = MODULE_NAMES[module]
        if mkey in cstatus:
            mstatus = cstatus[mkey]
        else:
            return None

        # This might be empty
        if not mstatus:
            return None

        # The first entry is the pipeline index
        if "pipeline_index" in mstatus:
            mindex = mstatus.pop("pipeline_index")
        else:
            mindex = 0

        # If the job was submitted but didn't get a job id
        if len(mstatus) == 1:
            job = mstatus[next(iter(mstatus))]
            if "job_id" not in job:
                job["job_id"] = 0
                mstatus[next(iter(mstatus))] = job

        # Create data frame
        mdf = pd.DataFrame(mstatus).T
        mdf["pipeline_index"] = mindex

        # If incomplete:
        if not mstatus:
            for col in mdf.columns:
                if col == "job_status":
                    mstatus[col] = "unsubmitted"
                else:
                    mstatus[col] = None
            mstatus = {mkey: mstatus}

        # If stats, add summary stat fields
        if self.stats:
            if "out_file" in mdf:
                mdf = self._add_stats(mdf)
            else:
                mdf = None

        # Otherwise, format the run status fields
        else:
            if "out_fpath" not in mdf and "dirout" not in mdf:
                mdf["file"] = None
            else:
                if "out_fpath" not in mdf:
                    mdf["file"] = mdf.apply(self.join_fpath, axis=1)
                else:
                    mdf["file"] = mdf["out_fpath"]

            # Add date
            if "time_end" not in mdf:
                mdf["date"] = mdf["time_start"]
                # mdf["date"] = mdf["file"].apply(self.find_date)
            else:
                mdf["date_submitted"] = mdf["time_submitted"]
                mdf["date_completed"] = mdf["time_end"]

            if "finput" not in mdf:
                mdf["finput"] = mdf["file"]

            if "runtime" not in mdf:
                mdf["runtime"] = "na"

        return mdf

    def check_entries(self, print_df, check):
        """Check for a specific status."""
        if check in FAILURE_STRINGS:
            print_df = print_df[print_df["job_status"] == "failed"]
        elif check in SUCCESS_STRINGS:
            print_df = print_df[print_df["job_status"] == "successful"]
        elif check in PENDING_STRINGS:
            print_df = print_df[print_df["job_status"] == "PD"]
        elif check in RUNNING_STRINGS:
            print_df = print_df[print_df["job_status"] == "R"]
        elif check in SUBMITTED_STRINGS:
            print_df = print_df[print_df["job_status"] == "submitted"]
        elif check in UNSUBMITTED_STRINGS:
            print_df = print_df[print_df["job_status"] == "unsubmitted"]
        else:
            print("Could not find status filter.")

        return print_df

    def check_index(self, df, sub_folder):
        """Check that log files for a given status index exist."""
        for i, row in df.iterrows():
            try:
                self.find_pid_dirs([sub_folder], row["job_id"])
            except FileNotFoundError:
                df["index"].iloc[i] = "NA"
        return df

    def checkout(self, logdir, pid, output="error"):
        """Print first 20 lines of an error or stdout log file."""
        # Find the appropriate files
        if output == "error":
            pattern = "*e"
            name = "Error"
            outs = glob(os.path.join(logdir, "stdout", pattern))
        else:
            pattern = "*o"
            name = "STDOUT"
            outs = glob(os.path.join(logdir, "stdout", pattern))

        # Find the target file
        try:
            log = [o for o in outs if str(pid) in o][0]
        except IndexError:
            print(Fore.RED + name + " log for job ID " + str(pid)
                  + " not found." + Style.RESET_ALL)
            return

        # Read each line in the log
        with open(log, "r") as file:
            lines = file.readlines()

        # Limit the number of lines if full_print is not set
        if not self.full_print:
            if len(lines) > 20:
                lines = lines[-20:]
                print("  \n   ...   \n")
        for line in lines:
            print(line)
        print(Fore.YELLOW + log + Style.RESET_ALL)

    def find_date(self, file):
        """Return the modification date of a file."""
        try:
            seconds = os.path.getmtime(str(file))
            date = dt.datetime.fromtimestamp(seconds)
            sdate = dt.datetime.strftime(date, "%Y-%m-%d %H:%M")
        except FileNotFoundError:
            sdate = "NA"
        return sdate

    def find_file(self, folder, file="config_pipeline.json"):
        """Check/return the config_pipeline.json file in the given directory."""
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            msg = (f"No {file} files found. If you were looking for nested "
                   "files, try running the with --walk option.")
            raise ValueError(Fore.RED + msg + Style.RESET_ALL)
        return path

    def find_files(self, folder, file="config_pipeline.json", pattern=None):
        """Walk the dirpath directories and find all file paths.

        Parameters
        ----------
        folder : str
            Path to root directory in which to search for files.
        file : str
            Name of a target files.
        pattern : str
            Pattern contained in target files.

        Returns
        -------
        list : list of file paths.
        """
        paths = []
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                if pattern:
                    if pattern in name:
                        paths.append(os.path.join(root, file))
                else:
                    if name == file:
                        paths.append(os.path.join(root, file))
            for name in dirs:
                if pattern:
                    if pattern in name:
                        paths.append(os.path.join(root, file))
                else:
                    if name == file:
                        paths.append(os.path.join(root, file))

        return paths

    def find_logs(self, folder):  # <------------------------------------------ Speed this up or use find_files
        """Find the log directory, assumes one per folder."""
        # If there is a log directory directly in this folder use that
        contents = glob(os.path.join(folder, "*"))
        possibles = [c for c in contents if "log" in c]

        if len(possibles) == 1:
            logdir = os.path.join(folder, possibles[0])
            return logdir

        # If that didn't work, check the current path
        if "/logs/" in folder:
            logdir = os.path.join(folder[:folder.index("logs")], "logs")
            return logdir

        # If that didn't work check the config files
        config_files = glob(os.path.join(folder, "*.json"))
        logdir = None
        for file in config_files:
            if not logdir:
                # The file might not open
                try:
                    config = json.load(open(file, "r"))
                except:  # <--------------------------------------------------- What would this/these be?
                    pass

                # The directory might be named differently
                try:
                    logdir = config["log_directory"]
                except KeyError:
                    logdir = config["directories"]["log_directory"]

        # Expand log directory
        if logdir:
            if logdir[0] == ".":
                logdir = logdir[2:]
                logdir = os.path.join(folder, logdir)
            logdir = os.path.expanduser(logdir)

        return logdir

    def find_outputs(self, folder):
        """Find the output directory, assumes one per folder."""
        # Check each json till you find it
        config_files = glob(os.path.join(folder, "*.json"))
        outdir = None

        if not config_files:
            print(Fore.RED + "No reV outputs found." + Style.RESET_ALL)
            return
        try:
            for file in config_files:
                if not outdir:
                    # The file might not open
                    try:
                        config = json.load(open(file, "r", encoding="utf-8"))
                    except:
                        pass

                    # The directory might be named differently
                    try:
                        outdir = config["directories"]["output_directory"]
                    except KeyError:
                        pass
        except:
            print("Could not find reV output directory")
            raise

        # Expand log directory
        if outdir:
            if outdir[0] == ".":
                outdir = outdir[2:]  # "it will have a / as well
                outdir = os.path.join(folder, outdir)
            outdir = os.path.expanduser(outdir)

        return outdir

    def find_pid_dirs(self, folders, target_pid):
        """Check the log files and find which folder contain the target pid."""
        pid_dirs = []
        for folder in folders:
            logs = glob(os.path.join(folder, "logs", "stdout", "*e"))
            for line in logs:
                file = os.path.basename(line)
                idx = file.rindex("_")
                pid = file[idx + 1:].replace(".e", "")
                if pid == str(target_pid):
                    pid_dirs.append(folder)

        if not pid_dirs:
            msg = "No log files found for pid {}".format(target_pid)
            raise FileNotFoundError(Fore.RED + msg + Style.RESET_ALL)

        return pid_dirs

    def infer_runtime(self, job, file):
        """Infer the runtime for a specific job (dictionary entry)."""
        # Find the run directory
        if "dirout" not in job:
            if ".gaps" in str(file):
                dirout = file.parent.parent
            else:
                dirout = file.parent
        else:
            dirout = job["dirout"]

        # Find all the output logs for this job id
        jobid = job["job_id"]
        logdir = self.find_logs(dirout)
        if logdir:
            stdout = os.path.join(logdir, "stdout")
            logpath = glob(os.path.join(stdout, "*{}*.o".format(jobid)))[0]

            # We can't get file creation in Linux
            with open(logpath, "r", encoding="utf-8") as file:
                lines = [line.replace("\n", "") for line in file.readlines()]

            time_lines = []
            for line in lines:
                if line.startswith(("INFO", "DEBUG", "ERROR", "WARNING")):
                    time_lines.append(line)
            date1 = time_lines[0].split()[2]
            date2 = time_lines[-1].split()[2]
            time1 = time_lines[0].split()[3][:8]
            time2 = time_lines[-1].split()[3][:8]

            # There might not be any values here
            try:
                time_string1 = " ".join([date1, time1])
                time_string2 = " ".join([date2, time2])
            except NameError:
                return "N/A"

            # Format these time strings into datetime objects
            dtime1 = dt.datetime.strptime(time_string1, "%Y-%m-%d %H:%M:%S")
            dtime2 = dt.datetime.strptime(time_string2, "%Y-%m-%d %H:%M:%S")

            # Take the difference
            minutes = round((dtime2 - dtime1).seconds / 60, 3)
        else:
            minutes = "N/A"

        return minutes

    def find_runtimes(self, status, file):
        """Find runtimes if missing from the main status json."""
        # Remove monitor pid
        if "monitor_pid" in status:
            _ = status.pop("monitor_pid")

        # For each entry, try to find or infer runtimes
        for module, entry in status.items():
            for label, job in entry.items():
                if isinstance(job, dict):
                    if "pipeline_index" != label:
                        if "job_id" in job and "runtime" not in job:
                            if "total_runtime" in job:
                                job["runtime"] = job["total_runtime"]
                            else:
                                job["runtime"] = self.infer_runtime(job, file)
                            status[module][label] = job
        return status

    def find_status(self, sub_folder):
        """Find the job status json."""
        # Two possible locations for status files
        sub_folder = Path(sub_folder)
        if sub_folder.joinpath(".gaps").exists():
            if any(sub_folder.joinpath(".gaps").glob("*status*")):
                sub_folder = sub_folder.joinpath(".gaps")

        # Find output directory
        try:
            files = list(sub_folder.glob("*.json"))
            file = [f for f in files if "_status.json" in f.name][0]
            job_files = list(sub_folder.glob("jobstatus*.json"))
        except IndexError:
            outdir = self.find_outputs(str(sub_folder))
            if outdir:
                files = glob(os.path.join(outdir, "*.json"))
                file = [f for f in files if "_status.json" in f]
                if not file:
                    return None, None
            else:
                return None, None

        # Get the status dictionary
        with open(file, "r", encoding="utf-8") as f:
            try:
                status = json.load(f)
            except json.decoder.JSONDecodeError:
                status = "Updating"

        # Update entries left over in job status files
        if job_files:
            for jfile in job_files:
                with open(jfile, "r", encoding="utf-8") as f:
                    try:
                        jstatus = json.load(f)
                    except json.decoder.JSONDecodeError:
                        pass
                jmodule = next(iter(jstatus))
                jname = next(iter(jstatus[jmodule]))
                if jmodule in status:
                    if jname in status[jmodule]:
                        if "job_id" not in status[jmodule][jname]:
                            job_id = "NA"
                        else:
                            job_id = status[jmodule][jname]["job_id"]
                        jstatus[jmodule][jname]["time_submitted"] = \
                            status[jmodule][jname]["time_submitted"]
                        jstatus[jmodule][jname]["job_id"] = job_id
                        status[jmodule][jname] = jstatus[jmodule][jname]

        # Fix artifacts
        if isinstance(status, dict):
            status = self.fix_status(status)

            # Fill in missing runtimes
            try:
                status = self.find_runtimes(status, file)
            except IndexError:
                pass

        return file, status

    def fix_status(self, status):
        """Fix problematic artifacts from older reV versions."""
        # Aggregation vs Supply-Curve-Aggregation
        if "aggregation" in status and "supply-curve-aggregation" in status:
            ag = status["aggregation"]
            scag = status["supply-curve-aggregation"]

            if len(scag) > len(ag):
                del status["aggregation"]
            else:
                status["supply-curve-aggregation"] = status["aggregation"]
                del status["aggregation"]
        elif "aggregation" in status:
            status["supply-curve-aggregation"] = status["aggregation"]
            del status["aggregation"]

        return status

    @property
    def folders(self):
        """Return appropriate folders."""
        if self.walk or self.error or self.out:
            folders = self.find_files(self.folder, file="logs")
            folders = [os.path.dirname(f) for f in folders]
            folders.sort()
        else:
            folders = [self.folder]
        return folders

    def join_fpath(self, row):
        """Join file directory and file name in a status dataframe."""
        check1 = "dirout" in row and "fout" in row
        check2 = row["dirout"] is not None
        check3 = row["fout"] is not None
        if check1 and check2 and check3:
            fpath = os.path.join(row["dirout"], row["fout"])
        else:
            fpath = "NA"
        return fpath

    @property
    def status_df(self):
        """Return full dataframe of status entries."""
        # Build data frame for one or more jobs
        if len(self.folders) > 1:
            df = self._run_parallel()
        else:
            df = self._run_single()
        return df

    @property
    def successful(self):
        """Quick check if all runs in run directory were successful."""
        df = self.status_df
        return all(df["job_status"] == "successful")

    def to_csv(self, df):
        """Save the outputs to a CSV."""
        # Make a destination path
        date = dt.datetime.today()
        stamp = dt.datetime.strftime(date, "%Y%m%d%H%M")
        if self.stats:
            dst = os.path.join(self.folder, f"rrlogs_{stamp}_stats.csv")
        else:
            dst = os.path.join(self.folder, f"rrlogs_{stamp}.csv")

        # Write file
        df.to_csv(dst, index=False)

    def _add_stat(self, fpath):
        """Returns summary statistics for a reV output file."""
        # Read in the mean capacity factor field for file
        ext = fpath.split(".")[-1]
        if ext == "csv":
            if self.field not in pd.read_csv(fpath, nrows=0).columns:
                raise KeyError(f"{self.field} not in {fpath}")
            df = pd.read_csv(
                fpath,
                usecols=["sc_point_gid", self.field]
            )
        elif ext == "h5":
            df = read_h5(fpath, self.field)
        else:
            raise NotImplementedError(f"Cannot summarize {ext} files.")

        # Calculate min, max, std, etc.
        return df.iloc[:, -1].describe()

    def _add_stats(self, mdf):
        """Add stats to a module status data frame."""
        # Only return for existing files (i.e., not incomplete or chunk_files)
        mdf["out_file"][pd.isnull(mdf["out_file"])] = "NaN"
        mdf["exists"] = mdf["out_file"].apply(os.path.exists)
        mdf = mdf[mdf["exists"]]
        print(mdf.columns)

        # Add stats if anything is left
        if mdf.shape[0] > 0:
            fpaths = mdf["out_file"]
            out = fpaths.apply(self._add_stat)
            mdf["fname"] = mdf["out_file"].parallel_apply(
                lambda x: os.path.basename(x)
            )
            mdf = mdf[["job_id", "fname"]]
            mdf = mdf.join(out)

        return mdf

    def _count_aus(self, df):
        """Count the AUs used in status data frame."""
        host = os.uname()[1]
        if host.startswith("k"):
            rate = 10
            host = "kestrel"
        elif host.startswith("e"):
            rate = 3
            host = "eagle"
        minutes = df["runtime"].sum()
        aus = (minutes / 60) * rate
        out = f"AUs ({host}, {rate}x) = {round(aus, 2):,}"
        print(out)

    def _run(self, args):
        """Print status and job pids for a single project directory."""
        # Unpack args
        folder, sub_folder, module, status, error, out = args

        # Expand folder path
        sub_folder = os.path.abspath(os.path.expanduser(sub_folder))

        # Convert module status to data frame
        df = self.build_dataframe(sub_folder, module)

        if isinstance(df, str) and df == "updating":
            print(Fore.YELLOW + f"\nStatus file updating for {sub_folder}"
                  + Style.RESET_ALL)
        elif df is None:
            print(Fore.RED + f"\nStatus file not found for {sub_folder}"
                  + Style.RESET_ALL)

        # This might return None
        else:
            logdir = self.find_logs(sub_folder)

            # Now return the requested return type
            if status:
                df = self.check_entries(df, status)

            if self.verbose:
                if not error and not out and df.shape[0] > 0 and not self.stats:
                    if not self.csv:
                        print_folder = os.path.relpath(sub_folder, folder)
                        self._status_print(df, print_folder, logdir)

                if self.stats:
                    print_folder = os.path.relpath(sub_folder, folder)
                    self._stat_print(df, print_folder)

            # If a specific status was requested
            if error or out:
                # Find the logging directoy
                if not logdir:
                    print(Fore.YELLOW
                          + "Could not find log directory."
                          + Style.RESET_ALL)
                    return

                # Print logs
                if error:
                    try:
                        pid = df["job_id"][df["index"] == int(error)].iloc[0]
                    except IndexError:
                        print(Fore.YELLOW + f"Error log for job id {error} "
                              "not yet available." + Style.RESET_ALL)
                        return
                    self.checkout(logdir, pid, output="error")
                if out:
                    try:
                        pid = df["job_id"][df["index"] == int(out)].iloc[0]
                    except IndexError:
                        print(Fore.YELLOW + f"Stdout log for job id {out} not "
                              "yet available." + Style.RESET_ALL)
                        return
                    self.checkout(logdir, pid, output="stdout")

        return df

    def _run_parallel(self):
        """Run if only multiple sub folders are present in main directory."""
        args = []
        for sub_folder in self.folders:
            arg = (self.folder, sub_folder, self.module, self.status,
                    self.error, self.out)
            args.append(arg)

        dfs = []
        with mp.Pool(mp.cpu_count() - 1) as pool:
            for out in pool.imap(self._run, args):
                dfs.append(out)

        df = pd.concat(dfs)
        df = df.dropna(axis=1, how="all")

        return df

    def _run_single(self):
        """Run if only one sub folder is present in main run directory."""
        args = (self.folder, self.folders[0], self.module, self.status,
                self.error, self.out)
        df = self._run(args)
        return df

    def _stat_print(self, df, print_folder):
        """Color the statistical portion of data frame and print."""
        # Filter for just stat columns
        df = df[["job_id", "fname", *list(STAT_COLORS)]]

        # Set stats to intuitive colors
        def color_column(values):
            name = values.name
            if name in STAT_COLORS:
                color = STAT_COLORS[name]
                values = values.apply(
                    lambda x: color + str(round(x, 4)) + Style.RESET_ALL
                )
            return values

        # Print containing folder name
        name = "\n" + Fore.CYAN + print_folder + Style.RESET_ALL + ":"

        # Color the table
        df = df.apply(color_column, axis=0)

        pdf = tabulate(df, showindex=False, headers=df.columns,
                       tablefmt="simple")
        print(f"{name}\n{pdf}")

    def _status_print(self, df, print_folder, logdir):
        """Color the status portion of data frame and print."""
        def color_string(string):
            if string in FAILURE_STRINGS:
                string = Fore.RED + string + Style.RESET_ALL
            elif string in SUCCESS_STRINGS:
                string = Fore.GREEN + string + Style.RESET_ALL
            elif string in RUNNING_STRINGS:
                string = Fore.BLUE + string + Style.RESET_ALL
            else:
                string = Fore.YELLOW + string + Style.RESET_ALL
            return string

        tcols = [col for col in PRINT_COLUMNS if col in df]
        df = df[tcols]

        name = "\n" + Fore.CYAN + print_folder + Style.RESET_ALL + ":"
        if "job_status" not in df:
            df.insert(2, "job_status", "unknown")
        df["job_status"] = df["job_status"].apply(color_string)
        pdf = tabulate(df, showindex=False, headers=df.columns,
                       tablefmt="simple")
        if not logdir:
            msg = "Could not find log directory."
            print(Fore.YELLOW + msg + Style.RESET_ALL)
        print(f"{name}\n{pdf}")

    def _to_hours(self, time_string):
        """Convert timestamp to number of hours."""
        if time_string != "N/A":
            hours, minutes, seconds = map(int, time_string.split(":"))
            hours += ((minutes / 60) + (seconds / 3_600))
        else:
            hours = pd.NA
        return hours

    def main(self):
        """Run the appropriate rrlogs functions for a folder."""
        # Turn off print states for certain flags
        if self.count_aus or self.csv:
            self.verbose = False

        # Run rrlogs for each
        df = self.status_df

        # Count AUs
        if self.count_aus:
            self._count_aus(df)

        # Write to file
        if self.csv:
            self.to_csv(df)


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", default=None, help=MODULE_HELP)
@click.option("--status", "-s", default=None, help=STATUS_HELP)
@click.option("--error", "-e", default=None, help=ERROR_HELP)
@click.option("--out", "-o", default=None, help=OUT_HELP)
@click.option("--walk", "-w", is_flag=True, help=WALK_HELP)
@click.option("--full_print", "-fp", is_flag=True, help=FULL_PRINT_HELP)
@click.option("--csv", "-c", is_flag=True, help=SAVE_HELP)
@click.option("--stats", "-st", is_flag=True, help=STATS_HELP)
@click.option("--field", "-fd", default=None, help=FIELD_HELP)
@click.option("--count_aus", "-au", is_flag=True, help=AU_HELP)
@click.option("--verbose", "-v", is_flag=True, default=True, help=VERBOSE_HELP)
def main(folder, module, status, error, out, walk, full_print, csv, stats,
         field, count_aus, verbose):
    r"""REVRUNS - Check Logs.

    Check log files of a reV run directory. Assumes reV run in pipeline.
    """
    rrlogs = RRLogs(
        folder=folder,
        module=module,
        status=status,
        error=error,
        out=out,
        walk=walk,
        full_print=full_print,
        csv=csv,
        stats=stats,
        field=field,
        count_aus=count_aus,
        verbose=verbose
    )
    rrlogs.main()


if __name__ == "__main__":
    folder = '/projects/rev/projects/ffi/fy24/rev/solar/test2/'
    sub_folder = folder
    error = None
    out = None
    walk = False
    module = None
    status = None
    full_print = True
    csv = False
    stats = False
    verbose = False
    field = None
    count_aus = False
    verbose = True
    self = RRLogs(folder, module, status, error, out, walk, full_print, csv,
                  stats, count_aus=count_aus, verbose=verbose)
    self.main()
