# -*- coding: utf-8 -*-
"""Debug CLI command.

This could turn into a more significant convenience wrapper for CliRunner. I'd
like to be able to quickly enter a debug session using a cli command with
pointers to configs. This would need to be used carefully, since some errors
require enormous memory and run times.

In this case I'm trying to recreate a transmission error on a chunked bespoke
hdf5 file.

Currently this assumes that folks named the configs directly after the module
which is of course not usually true. It should work for pipelines though.

For VSCode, enable the debugging console, it helps a lot:
    - https://code.visualstudio.com/docs/python/debugging


Author: twillia2
Date: Fri Aug 19 13:21:39 MDT 2022
"""
import json

from pathlib import Path

import click
import pandas as pd
import reV

from click.core import Context
from click.testing import CliRunner
from reV.cli import main as base_cli


class RRDebug:
    """Methods for setting up and debugging sample reV runs from a CLI."""

    def __init__(self, module="pipeline", rundir="."):
        """Initialize RRDebug object.
        
        Parameters
        ----------
        cmd : str
            A string representation of a reV cli command. (e.g. "reV -c 
            config_pipeline.json pipeline")
        """
        self.rundir = Path(rundir).absolute().expanduser()
        self.module = module
        self._setup()

    def __repr__(self):
        """Return reprentation string for RRDebug object."""
        name = self.__class__.__name__
        address = hex(id(self))
        attrs = [f"  {k}={v}\n" for k, v in self.__dict__.items()]
        return f"<{name} object at {address}>\n{''.join(attrs)}"

    def adjust_inputs(self):
        """Adjust inputs, subset project points."""
        # Get the main config used for module
        path = self.rundir.joinpath(f"config_{self.module}.json")
        main_config = json.load(open(path))

        # If the config file is a pipeline, adjust each config in pipeline
        if self.module == "pipeline":
            pipeline = main_config["pipeline"]
            new_pipeline = []
            for entry in pipeline:
                module = list(entry)[0]
                path = list(entry.values())[0]
                new_pipeline.append({module: f"./config_{module}.json"})
                config = json.load(open(path))
                self.copy_config(config, module)
            main_config["pipeline"] = new_pipeline

        # Now write the main config
        self.copy_config(main_config, self.module)

    def copy_config(self, config, module):
        """Copy module config to debug dir."""
        # Make paths absolute
        for key, value in config.items():
            if isinstance(value, str):
                if value.startswith("./"):
                    path = str(self.rundir.joinpath(value))
                    config[key] = path

        # Configure to run locally
        if "execution_control" in config:
            config["execution_control"] = {"option": "local"}

        # Make sure the logging is going into the debug dir
        if "log_directory" in config:
            config["log_directory"] = str(self.dbg_dir.joinpath("logs"))

        # Might as well turn debug logging on
        if "log_level" in config:
            config["log_level"] = "DEBUG"

        # Replace project points
        if "project_points" in config and config["project_points"] is not None:
            ppath = Path(config["project_points"])
            pdst = self.dbg_dir.joinpath("project_points_sample.csv")
            config["project_points"] = str(pdst)
            if not pdst.exists():
                pp = pd.read_csv(ppath, nrows=25)
                pp.to_csv(pdst, index=False)

        # Write to debug dir
        dst = self.dbg_dir.joinpath(f"config_{module}.json")
        self._write_config(config, dst)

    @property
    def args(self):
        """Return arguments for cli command."""
        return self.dbg_cmd.split()[1:]

    @property
    def base_cli(self):
        """Return the appropriate reV click object for a given cli command."""
        # Only doing reV for now
        return base_cli

    @property
    def cli(self):
        """Return the appropriate reV click object for a given cli command."""
        # Find the command and config
        module = self.cmd.split()[-1]
        module = module.replace("-", "_")
        return reV.__dict__[module].__dict__[f"cli_{module}"].from_config

    @property
    def config_path(self):
        """Return the config path from a command."""
        return Path(self.cmd.split("-c ")[1:][0].split()[0])

    def run(self):
        """Run a CLI command in a Python session.

        Parameters
        ----------
        cmd : str
            A string representation of a reV cli command line interface call.
        """
        # Build runner object
        runner = CliRunner()

        # Using base command
        cli = self.base_cli
        args = self.cmd.split()[1:]
        out = runner.invoke(cli, args=args)

        # This just needs to get past submission before raising an error
        assert out.exit_code == 0, f"{self.cmd} failed"

    def write_script(self):
        """Write an implementation of `RRDebug.run` to a python file."""
        py_path = self.dbg_dir.joinpath("debug.py")
        lines = self._header + self._imports + self._body + self._call
        lines = [line + "\n" for line in lines]
        with open(py_path, "w") as file:
            file.writelines(lines)

    @property
    def _body(self):
        """Return body for script."""
        body = [
            "",
            "",
            "def main():",
            "    # Build runner and debug objects",
            "    runner = CliRunner()",
            "", 
            "    # Using base reV with arguments command",
            f"    args = {list(self.args)}",
            "    _ = runner.invoke(cli, args=args, catch_exceptions=False)"
        ]
        return body

    @property
    def _call(self):
        """Return body for script."""
        call = [
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
        return call

    @property
    def _header(self):
        """Return header for script."""
        header = [
            "# -*- coding: utf-8 -*-",
            '"""RRDebug Script.',
            "",
            "Use this script with a debugger and break points to catch reV.",
            "errors from the following CLI command:",
            "",
            f"`{self.cmd}`",
            '"""',
        ]
        return header

    @property
    def _imports(self):
        """Return body for script."""
        imports = [
            "import os",
            "",
            "from click.testing import CliRunner",
            "from reV.cli import main as cli",
            "",
            f"os.chdir('{self.dbg_dir}')"
        ]
        return imports

    def _setup(self):
        """Setup debug directory, configs, and commands."""
        # Build command
        if reV.__version__ < "0.8.0":
            self.cmd = f"reV -c config_{self.module}.json {self.module}"
        else:
            self.cmd = f"reV {self.module} -c config_{self.module}.json"

        # Build paths and make debug directory
        run_dir = self.config_path.parent.absolute().expanduser()
        self.dbg_dir = run_dir.joinpath("debug")
        self.dbg_dir.mkdir(exist_ok=True)

        # Create a debug command for the python script
        self.dbg_config_path = self.dbg_dir.joinpath(self.config_path.name)
        self.dbg_cmd = self.cmd.replace(
            str(self.config_path),
            str(self.dbg_config_path)
        )

    def _write_config(self, config, path):
        # Write the appropriate config for a given module to debug dir
        with open(path, "w") as file:
            file.write(json.dumps(config, indent=4))

    def main(self):
        """Copy needed inputs and write python script."""
        self.adjust_inputs()
        self.write_script()


@click.command()
@click.argument("directory", default=".")
def main(directory):
    """RRDebug - Create debugging files.
    
    Creates a debug directory with sample data and a python script that
    recreates a cli command.
    """
    rrdb = RRDebug(rundir=directory)
    rrdb.main()
