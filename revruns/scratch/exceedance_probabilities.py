# -*- coding: utf-8 -*-
"""Module Name.

Author: twillia2
Date: Wed Oct 30 11:34:24 MDT 2024
"""
from pathlib import Path


HOME = Path("/Users/twillia2/github/revruns/revruns/scratch")


class Template:
    """A docstring."""

    def __init__(self):
        """Initialize a Template object."""

    def __repr__(self):
        """Return a Template object representation string."""
        address = hex(id(self))
        name = self.__class__.__name__
        msgs = [f"\n   {k}={v}" for k, v in self.__dict__.items()]
        msg = " ".join(msgs)
        return f"<{name} object at {address}>: {msg}"


def main():
    """A docstring."""


if __name__ == "__main__":
    pass
