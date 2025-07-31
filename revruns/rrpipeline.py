# -*- coding: utf-8 -*-
"""Run reV pipeline configs.

Created on Sat Sep  5 17:03:21 2020

@author: twillia2
"""
import click
import json
import os
import shlex
import subprocess as sp

from pathlib import Path

from colorama import Fore, Style
from rex.utilities.execution import SubprocessManager

from revruns.rrlogs import RRLogs
from revruns import REV_VERSION


DIR_HELP = ("The directory containing one or more config_pipeline.json "
            "files. Defaults to current directory. (str)")
WALK_HELP = ("Walk the directory structure and run all config_pipeline.json "
             "files. (boolean)")
FILE_HELP = "The filename of the configuration file to search for. (str)"
PRINT_HELP = "Print the path to all pipeline configs found instead. (boolean)"


def check_version_greater_than(version1, version2):
    """Check if a module's version string is greater or equal to anothers."""
    v1 = sum([int(p) for p in version1.split(".")])
    v2 = sum([int(p) for p in version2.split(".")])
    return v1 >= v2


def check_status(pdir):
    """Check if a status file exists and is fully successful."""
    successful = False
    rrlogs = RRLogs()

    try:
        status_file, status = rrlogs.find_status(pdir)
        status_dir = os.path.dirname(status_file)
        pipeline_file = os.path.join(status_dir, "config_pipeline.json")
        pipeline = json.load(open(pipeline_file, "r"))
    except (OSError, TypeError):
        status = None

    if status:
        modules = [list(p.keys())[0] for p in pipeline["pipeline"]]
        if len(status) == len(modules):
            statuses = []
            for m in modules:
                entry = status[m]
                for key in list(entry.keys())[1:]:
                    try:
                        jobstatus = entry[key]["job_status"]
                    except (IndexError, KeyError):
                        jobstatus = "not submitted"
                    statuses.append(jobstatus)
            if all([s == "successful" for s in statuses]):
                successful = True

    return successful


@click.command()
@click.option("--dirpath", "-d", default=".", help=DIR_HELP)
@click.option("--walk", "-w", is_flag=True, help=WALK_HELP)
@click.option("--file", "-f", default="config_pipeline.json", help=FILE_HELP)
@click.option("--print_paths", "-p", is_flag=True, help=PRINT_HELP)
def rrpipeline(dirpath, walk, file, print_paths):
    """Run one or all reV pipelines in a directory."""
    dirpath = os.path.expanduser(dirpath)
    dirpath = os.path.abspath(dirpath)
    rrlogs = RRLogs()
    print(Fore.CYAN + f"Running rrpipeline for {dirpath}..." + Style.RESET_ALL)
    if walk:
        config_paths = rrlogs.find_files(dirpath, file)

        # Remove top level pipeline for batch runs
        config_paths = [
            path for path in config_paths if str(Path(path).parent) != dirpath
        ]
    else:
        config_paths = [rrlogs.find_file(dirpath, file)]

    config_paths.sort()

    for path in config_paths:
        pdir = os.path.dirname(path)
        successful = check_status(pdir)
        rpath = os.path.join(".", os.path.relpath(path, dirpath))
        if print_paths:
            ppath = Fore.CYAN + rpath + ": " + Style.RESET_ALL
            if successful:
                pstatus = Fore.GREEN + "successful." + Style.RESET_ALL
            else:
                pstatus = Fore.RED + "not successful." + Style.RESET_ALL
            print(ppath + pstatus)
        else:
            if not successful:
                print(Fore.CYAN + "Submitting " + rpath + "..."
                      + Style.RESET_ALL)
                
                if check_version_greater_than(REV_VERSION, "0.8.0"):
                    cmd = f"reV pipeline -c {path} --monitor --background"
                else:
                    cmd = f"reV -c {path} pipeline --monitor --background"
                cmd = shlex.split(cmd)
                output = os.path.join(os.path.dirname(path), "pipeline.out")
                with sp.Popen(
                    cmd,
                    stdout=open(output, "w", encoding="utf-8"),
                    stderr=open(output, "w", encoding="utf-8")
                    # preexec_fn=os.setpgrp
                ) as process:
                    if process.returncode == 1:
                        raise OSError(
                            "Submission failed: check {}".format(output)
                        )
                    SubprocessManager.submit(cmd, background=True,
                                            background_stdout=False)

    return config_paths
