# -*- coding: utf-8 -*-
"""Install revruns.

If installing in conda environment, you may have to set some variables first:

export CPLUS_INCLUDE_PATH="/pathtoenv/include/gdal.h" 
export C_INCLUDE_PATH="/pathtoenv/include/gdal.h"
"""
import subprocess as sp

from setuptools import setup

# from Cython.Build import cythonize


def get_gdal_version():
    """Return system GDAL version."""
    process = sp.Popen(
        ["gdal-config", "--version"],
        stdout=sp.PIPE,
        stderr=sp.PIPE
    )
    sto, ste = process.communicate()
    if ste:
        raise OSError("GDAL is causing problems again. Make sure you can run "
                      "'gdal-config --version' successfully in your terminal")
    version = sto.decode().replace("\n", "")
    return version


def get_requirements():
    """Get requirements and update gdal version number."""
    with open("requirements.txt", encoding="utf-8") as file:
        reqs = file.readlines()
    gdal_version = get_gdal_version()
    gdal_line = [req for req in reqs if req.startswith("pygdal")][0]
    gdal_line = gdal_line[:-1]
    reqs = [req for req in reqs if not req.startswith("pygdal")]
    gdal_line = f"{gdal_line}=={gdal_version}.*\n"
    reqs.append(gdal_line)
    return reqs


setup(
    name='revruns',
    version='0.0.2',
    packages=['revruns'],
    description=("Functions and CLIs that to help to configure, run, "
                 "and check outputs for NREL's Renewable Energy Technical "
                 "Potential Model (reV)."),
    author="Travis Williams",
    author_email="travis.williams@nrel.gov",
    # ext_modules=cythonize("revruns/cython_compute.pyx"),
    install_requires=get_requirements(),
    include_package_data=True,
    package_data={
        "data": [
            "*"
        ]
    },
    entry_points={
        "console_scripts":
            [
                "rrbatch_collect = revruns.rrbatch_collect:main",
                "rrbatch_hack = revruns.rrbatch_hack:main",
                "rrbatch_logs = revruns.rrbatch_logs:main",
                "rrcheck = revruns.rrcheck:main",
                "rrdebug = revruns.rrdebug:main",
                "rrconnections = revruns.rrconnections:main",
                "rrcomposite = revruns.rrcomposite:main",
                "rrerun = revruns.rrerun:main",
                "rreset = revruns.rreset:main",
                "rrexclusion = revruns.rrexclusion:main",
                "rrgraphs = revruns.rrgraphs:main",
                "rrgeoref = revruns.rrgeoref:main",
                "rrlogs = revruns.rrlogs:main",
                "rrlist = revruns.rrlist:main",
                "rrpipeline = revruns.rrpipeline:rrpipeline",
                "rrpoints = revruns.rrpoints:main",
                "rrprofiles = revruns.rrprofiles:main",
                "rrshape = revruns.rrshape:main",
                "rrsetup = revruns.rrsetup:main",
                "rraster = revruns.rraster:main",
                "rrtemplates = revruns.rrtemplates:main"
            ]
        }
    )
