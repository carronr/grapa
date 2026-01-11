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

-  Open and merge a wide range of experimental data.
   Data import possible from clipboard.
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


Grapa can also be used in scripts or own code. To help you to get
started, the GUI offers a checkbox "Commands in console" right above
the console. Use the GUI and learn from the printouts.

Grapa is developed using python 3.13 and matplotlib 3.10, and should be
backwards-compatible down to python 3.7 and matplotlib 1.5.



==================
License
==================

MIT License



=======================
Details of the features
=======================

The grapa packages comes with a graphical user interface (GUI), which allows
handling most features of the package. Many matplotlib plotting methods are
supported, as well as insets and subplots. Using grapa in python code allows
further customization of the produced graphs.

The central object of the package is the Graph object. A graph is stored in a
Graph object. The Graph class can be thought as a combination of a dict
storing plotting information (axes, text annotations, etc.) together with a
list of Curves which stores the data and the corresponding plotting information
(color, aspect, etc.).
The Graph.plot() method returns the produced figure and axes for further
modifications. Existing figures and axes can as well be provided.
List of relevant keywords: see GUI, or file keywordsdata_graph.txt


A Curve object contains the data usually accessed as a pair of vectors,
as well as a dict-like metadata storage, including plotting information.
Child classes of Curve can be implemented in order to provide dedicated
data analysis tools.
List of relevant keywords: see GUI, or file keywordsdata_curve.txt
Every Curve type provides dedicated functions (e.g. for analysis or plotting)
as well as specific data visualizations accessible from a GUI drop-down menu.
Default Curve type for plotting purpose:
Curve_Subplot, Curve_Inset, Curve_Image

Other Curve types are provided for analysis of scientific data:
EQE, J-V, Jsc-Voc pairs, Spectrum, SIMS, C-V, C-f, TRPL, XRF


Scripts are provided:
- Load all files from one folder
- JV data fit, cell-by-cell, or cell separately
- JV sample maps
- Boxplots, both generic and specific behavior for JV data
- Processing of C-V data temperature series
- Processing of C-f data temperature series
- Correlation


Configuration file: a set of user preferences can be stored in a configuration
file and indicated to the Graph object at creation. A personalized
configuration file can also be provided to the GUI when using a application
launcher as a second argument (it will be retrieved using sys.argv[1])
Whenever needed the software will query for provided values, and use default
settings if no suitable keyword is found.



====================
Create a file parser
====================

To create a file parser you will need to create inside folde "datatypes"
a python file named for example graphMyfile.py (must start with "graph",
with a small g).
This file must contains a class named "GraphMyfile" (capital G, same ending as
file name) that inherits from class Graph.
The class must contain at least 1 attribute and 2 methods:

- CURVE: a string identifier, for recognition within grapa.

- isFileReadable(cls, filename, fileext, line1="", line2="", line3="", \*\*kwargs)

  Must return True is the file can be opened using your class, otherwise False

- readDataFromFile(self, attributes, \*\*kwargs)

Actually opens the file, which can be accessed from self.filename.
attributes is a dict which can be given to the constructor of the Curves.
Example: see files howto/example_sinx_x.txt, graphMydata.py

You may also want to create your own Curve type, for example to provide
customised analysis or visualisation functions inside the GUI. For this, you
have to create in folder "datatypes" a file named for example
"curveMycurve.py" (file name must start with "curve" with a small c).
The file must contain a class named "CurveMycurve" (capital C, same ending as
the file name) inheriting from class Curve.
An example is provided in file howto/curveMycurve.py.



==================
Questions?
==================

Author: Romain Carron (Empa, Laboratory for Thin Films and
Photovoltaics)
