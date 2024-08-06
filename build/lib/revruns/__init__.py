"""reV runs.

A set of utilities and command line interfaces that help to setup and run reV.
"""
from importlib.metadata import version, PackageNotFoundError

from .paths import Paths


try:
    REV_VERSION = version("NREL-reV")
except PackageNotFoundError:
    REV_VERSION = None
