# -*- coding: utf-8 -*-
"""Reset reV status files after moving a project to a new directory.

Author: twillia2
Date: Fri Jun 23 14:09:05 MDT 2023
"""
import click
import json
import os
import shutil

from pathlib import Path

from colorama import Fore, Style

from revruns.rrlogs import RRLogs


OLD_DIR_HELP = ("Path to folder from which a reV run was moved. (str)")
NEW_DIR_HELP = ("Path to folder into which a reV run was moved. Defaults to "
                "current directory. (str)")
ALL_HELP = ("Update all paths found in status file. Defaults to updating only "
            "the 'dirout' entries. (boolean)")
REVERT_HELP = ("Revert reset status file to the previous status file. "
               "(boolean)")
WALK_HELP = ("Walk the given directory structure and update the status of "
             "all jobs found. (boolean)")


class RReset:

    def __init__(self, old_dir, new_dir=".", all_paths=False, revert=False,
                 walk=False):
        """Initialize an RRLogs object."""
        self.pwd = Path(".").absolute()
        if old_dir:
            old_dir = Path(old_dir)
        self.old_dir = old_dir
        self.new_dir = Path(new_dir).absolute()
        self.all_paths = all_paths
        self.revert = revert
        self.walk = walk

    def __repr__(self):
        """Return RRLogs object representation string."""
        attrs = ", ".join(f"{k}={v}" for k,v in self.__dict__.items())
        return f"<RReset object: {attrs}>"

    def copy_status(self, file):
        """Create a backup copy of the status file."""
        dst = file.parent.joinpath(file.name.replace(".json", "_old.json"))
        if not dst.exists():
            shutil.copy(file, dst)

    def reset(self, file):
        """Reset paths in a status file."""
        # Alert user to found status file
        msg = (f"Updating {file.name}...")
        print(Fore.CYAN + msg + Style.RESET_ALL)

        # Reset all or just one path
        if self.all_paths:
            keys = ["dirout", "gen_fpath", "finput"]
        else:
            keys = ["dirout"]

        # Read in status file
        config = json.load(open(file))
        for module, entry in config.items():
            for job, status in entry.items():
                if isinstance(status, dict):
                    for key in keys:
                        # Replace items in file input entry
                        if "finput" in status and key == "finput":
                            old = status["finput"]
                            new = old.copy()
                            for i, path in enumerate(old):
                                if isinstance(path, str):
                                    new_path = self.reset_path(path)
                                    new[i] = new_path
                            config[module][job]["finput"] = new

                        # Replace items in 'dirout' and/or 'gen_fpath' entries
                        elif key in status:
                            old = status[key]
                            new = self.reset_path(old)
                            config[module][job][key] = new


        with open(file, "w") as f:
            f.write(json.dumps(config, indent=4))

    def reset_path(self, path):
        """Reset a single path."""
        # Get user-defined paths
        old = str(self.old_dir)
        new = str(self.new_dir)

        # Remove file system components (how to make this universal?)
        old = old.replace("/vast", "").replace("/lustre/eaglefs", "")
        new = new.replace("/vast", "").replace("/lustre/eaglefs", "")

        # Replace patterns
        npath = path.replace(old, new)

        return npath

    def revert_status(self, file):
        """Replace updated status file with old status file."""
        old = file.parent.joinpath(file.name.replace(".json", "_old.json"))
        if old.exists():
            msg = (f"Reverting {file.name}...")
            print(Fore.CYAN + msg + Style.RESET_ALL)
            os.remove(file)
            shutil.move(old, file)
        else:
            msg = (f"No old status found to use to revert.")
            print(Fore.RED + msg + Style.RESET_ALL)

    def main(self):
        """Reset status file or files associated with a reV project."""
        # If walk find all project directories with a ...?
        if self.walk:
            statuses = list(self.pwd.rglob("*status.json"))
        else:
            statuses = list(self.pwd.glob("*status.json"))

        # Run rreset for each status file found
        if len(statuses) == 0 and self.walk is None:
            msg = (f"No in status files. Try running with the --walk option\n")
            print(Fore.RED + msg + Style.RESET_ALL)
        elif len(statuses) == 0 and self.walk is not None:
            msg = (f"No in status files.\n")
            print(Fore.RED + msg + Style.RESET_ALL)
        else:
            for file in statuses:
                if self.revert:
                    self.revert_status(file)
                else:
                    self.copy_status(file)
                    self.reset(file)


@click.command()
@click.option("--old_dir", "-o", default=None, help=OLD_DIR_HELP)
@click.option("--new_dir", "-n", default=".", help=NEW_DIR_HELP)
@click.option("--all_paths", "-a", is_flag=True, help=ALL_HELP)
@click.option("--revert", "-r", is_flag=True, help=REVERT_HELP)
@click.option("--walk", "-w", is_flag=True, help=WALK_HELP)
def main(old_dir, new_dir, all_paths, revert, walk):
    """REVRUNS - Reset reV Status Files.

    Replace an old directory path in the status files of a moved reV project
    with the new directory path. New directory defaults to current directory.
    """
    rrlogs = RReset(old_dir, new_dir, all_paths, revert, walk)
    rrlogs.main()


if __name__ == "__main__":
    old_dir = "/shared-projects/rev/projects/hfto/fy23/atb/pv_utility/"
    new_dir = "/shared-projects/rev/projects/hfto/fy23/rev/atb/pv_utility/"
    self = RReset(old_dir=old_dir, all_paths=True, revert=True, walk=True)
    self.main()
