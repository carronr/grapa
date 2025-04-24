Grapa
=====

DOI: 10.5281/zenodo.1164571

https://carronr.github.io/grapa/

================
Description
================

Grapa is a python package providing a graphical interface and the
underlying code dedicated to the visualization, analysis and
presentation of scientific data, with a focus on photovoltaic research.

A wide range of data formats can loaded by default. The produced graphs
are saved both as graphical objects and as text files for later
modifications.

The data analysis functions currently focus on photovoltaics and
semiconductor material science. Advanced analysis or fitting functions
are notably provided for the following characterization techniques: EQE,
JV, C-V, C-f, TRPL, SIMS, reflectance, transmittance (not comprehensive
list).

The software has extended capabilities for advanced plotting including
subplots and insets, and can be used for creating high-quality figures
for scientific publications.

Last, the user can add to the software and to the graphical interface
his own specific data loading functions as well new data types and
analysis functions, with no modification of the existing code.

Grapa stands for "GRAphing and Photovoltaics Analysis".

Cheers!


==================
Features overview
==================

-  Open and merge a wide range of experimental data. Data import
   possible from clipboard.
-  Graphical data presentation and graph customisation.
-  Graph export in various file formats with user-defined dpi.
   Additionally create a .txt file containing the raw data for later
   modifications.
-  The supported image formats are the one supported by the savefig
   function of matplotlib. .emf is also supported, provided a path to an
   inkscape executable is provided in the config.txt file.
-  Drop-down menu to switch between data-relevant views (e.g. log-view
   for JV data, Tauc plot for EQE)
-  Drop-down menu to switch between linear or logarithmic data
   visualisation.
-  A data picker allows selecting and storing chosen data points.
-  Can colorize a graph using color gradients.
-  Some data types can be fitted. The parameters of the fitted curves
   can be modified afterwards.
-  Data can be loaded from other files or from clipboard, then converted
   to the desired data type to benefit from data visualisation and
   processing and fitting options.


==================
Installation
==================

There are several options to install grapa. The easiest is certainly:

.. code-block:: python

   pip install grapa

Alternatively, the source can be retrieved from the development page
on github (https://github.com/carronr/grapa_). Download the latest
release and place its content in a folder named “grapa” somewhere on
your hard drive, and can be run with your favorite python distribution.

The software graphical user interface GUI can be started by executing
the following lines:


.. code-block:: python

   import grapa
   grapa.grapa()


Grapa can also be used in scripts. Just above the console, a
checkbox "Commands in console" should help you to get started.

Grapa is developed using python 3.13 and matplotlib 3.10, and should be
backwards-compatible down to python 3.4 and matplotlib 1.5.



==================
License
==================

MIT License



==================
Questions?
==================


See manual.pdf in folder /manual

Author: Romain Carron (Empa, Laboratory for Thin Films and
Photovoltaics)
