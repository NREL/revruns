#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking all batch run logs for success.

To Do:
    - use the gen_config to determime which module logs to check.

Created on Wed Feb 19 07:52:30 2020

@author: twillia2
"""
from glob import glob
from colorama import Fore, Style

import json
import os
import click


FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
MODULE_HELP = ("The reV module logs to check: generation, collect, "
               "multi-year, aggregation, supply-curve, or rep-profiles")

MODULE_DICT = {"gen": "generation",
               "collect": "collect",
               "multi-year": "multi-year",
               "aggregation": "aggregation",
               "supply-curve": "supply-curve",
               "rep-profiles": "rep-profiles"}
CONFIG_DICT = {"gen": "config_gen.json",
               "collect": "config_collect.json",
               "multi-year": "config_multi-year.json",
               "aggregation": "config_ag.son",
               "supply-curve": "config_supply-curve.json",
               "rep-profiles": "config_rep-profiles.json"}


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", default="gen", help=MODULE_HELP)
def main(folder, module):
    """
    revruns Batch Logs

    Check all logs for a reV module in a batched run directory.

    example:

        rrbatch_logs -f "." -m generation

    sample arguments:
        folder = "/projects/rev/new_projects/sergei_doubleday/5min"
        module = "gen"
    """

    # Open the gen config file to determine batch names
    os.chdir(folder)
    with open(CONFIG_DICT["gen"], "r") as file:
        config = json.load(file)
    tech = config["technology"]
    module_name = MODULE_DICT[module]

    # List all batch folders and check that they exist
    batches = glob("{}_*".format(tech))
    try:
        assert batches
    except AssertionError:
        print(Fore.RED + "No batch runs found." + Style.RESET_ALL)
        return

    # Check for "non-successes"
    failures = 0
    for batch in batches:
        stat_file = "{0}/{0}_status.json".format(batch)

        # Check that the file 
        try:
            with open(stat_file, "r") as file:
                log = json.load(file)
                genstat = log[module_name]
                runkeys = [k for k in genstat.keys() if batch in k]
                
                try:
                    assert runkeys
                except AssertionError:
                    print(Fore.RED + "No status found for " + batch +
                          Style.RESET_ALL)
                    failures += 1

                for k in runkeys:
                    try:
                        status = genstat[k]["job_status"]
                        assert status == "successful"  # <--------------------- A job_status for the last module of a pipeline might not update to successful, even if it is.
                    except AssertionError:
                        failures += 1
                        if status == "submitted":
                            print("Job '" + k + "' may or may not be fine. " +
                                  "Status: " + Fore.YELLOW + status)
                        else:
                            print("Job status '" + k + "': " + Fore.RED +
                                  status + ".")
                        print(Style.RESET_ALL)
        except FileNotFoundError:
            failures += 1
            print(Fore.RED)
            print("No log file found for '" + batch + "'.")
            print(Style.RESET_ALL)

    # Print final count. What else would be useful?
    if failures == 0:
        print(Fore.GREEN)
    else:
        print(Fore.RED)
    print("Logs checked with {} incompletions.".format(failures))

    # Reset terminal colors
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
