pd-combine: Combine Output Files
=================================

The ``pd-combine`` tool combines NetCDF output files from Snapy simulations.

Command Line Usage
------------------

.. code-block:: bash

    pd-combine <output_ids> [options]

Arguments:
    * ``output_ids``: Comma-separated list of output IDs to combine (e.g., "1,2,3")

Options:
    * ``-d, --dir``: Directory of the simulation to combine (default: current directory)
    * ``-o, --output``: Combined output name (default: "main")
    * ``--no-remove``: Do not remove original files
    * ``--no-merge``: Do not merge different fields

Python API
----------

.. module:: snapy.api.pd_combine

Functions
~~~~~~~~~

.. function:: CombineTimeseries(case: str, field: str, stamps: list, path: str = "./", remove: bool = False) -> None

   Concatenate output field across time stamps.

   :param case: Case name
   :type case: str
   :param field: Field name
   :type field: str
   :param stamps: List of time stamps
   :type stamps: list
   :param path: Path to output files
   :type path: str, optional
   :param remove: Remove original files after combining
   :type remove: bool, optional

.. function:: CombineFields(case: str, fields: str, name: str, path: str = "./") -> None

   Combine multiple output fields into a single file.

   :param case: Case name
   :type case: str
   :param fields: Comma-separated field IDs
   :type fields: str
   :param name: Output name
   :type name: str
   :param path: Path to output files
   :type path: str, optional

.. function:: ParseOutputFields(path: str) -> tuple

   Parse output fields from a directory.

   :param path: Path to output directory
   :type path: str
   :return: Tuple of (cases, fields, stamps)
   :rtype: tuple

.. function:: CombineFITS(case: str, output: str, path: str = "./", remove: bool = False) -> str

   Combine FITS output files.

   :param case: Case name
   :type case: str
   :param output: Output name
   :type output: str
   :param path: Path to output files
   :type path: str, optional
   :param remove: Remove original files after combining
   :type remove: bool, optional
   :return: Output FITS filename
   :rtype: str

Examples
--------

Combine outputs 1, 2, and 3::

    pd-combine 1,2,3

Combine with custom output name::

    pd-combine 1,2,3 -o combined

Combine without removing original files::

    pd-combine 1,2,3 --no-remove

Using the Python API::

    from snapy.api.pd_combine import CombineTimeseries, CombineFields
    
    # Combine time series
    CombineTimeseries("mycase", "out1", ["00001", "00002", "00003"])
    
    # Combine fields
    CombineFields("mycase", "1,2,3", "main")

Output Files
------------

The tool expects NetCDF files with the naming pattern::

    <case>.<field>.<stamp>.nc

For example::

    jupiter.out1.00001.nc
    jupiter.out1.00002.nc
    jupiter.out2.00001.nc

The combined output will be::

    <case>-<name>.nc

For example::

    jupiter-main.nc
