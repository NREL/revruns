"""reV runs.

A set of utilities and command line interfaces that help to setup and run reV.
"""
import importlib.metadata

from .paths import Paths


REV_VERSION = importlib.metadata.version("NREL-reV")
