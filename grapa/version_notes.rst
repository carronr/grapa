GRAPA: GRAphing and Photovoltaics Analysis
===========================================
Empa TFPV data reading software

** KNOWN BUGS **

- A few are there for sure, please announce them - romain.carron@empa.ch
- CurveTRPL fit with fixed tau parameters.
- Bug can prevent ERE to complete (to do: find a dataset that reproduces the error)





**Version 0.7.0.0

Release 23.04.2025

*Additions*

- Documentation of Grapa code, using sphinx.
- Added a conditional formatting mechanisms, available in the GUI just under the Template and Colorize. The behavior follows the pattenr "for each curve in graph, if [property] [==, contains etc.] [value], then [property] = [value]. Use "linestyle" "none" to hide a curve.
- Added a mechanisms to be able to easily provide fitting functionalities to specific Curve types, in a customizable manner (utils.py class FitHandler). The parametrization anc hcoice of pre-set fit setting are outsourced to an external configuration file, leaving much more freedom to the user. Should be rather easy to reuse for other use cases.
- Added the fit mechanisms into the CurveMCA used for XRF .mca data fit. A number of pre-defined fit settings are pre-configured in file XRF_fitparameters.txt, to suit the Abt207 XRF instrument and typical data. X-ray energy values can be specified: the Curve will lookup the correct channels using internal channel-kev conversion.
- Rewrote the xml parser into something that may be useful in the future. It can be configured from an external text file to tailor to specific formats. Hopefully the approach is generic enough to be useful.
- Added possibility to read data and metadata from xrdml files, using the xml parser.
- Added a Curve type, Curve XRD. Functionalities are extremely basic - please use dedicated software for anything more than plotting and fitting with gaussian (see new mechanisms above, and new configuration file XRD_fitparameters.txt).
- Curve TRPL: Added action, fit piece-wise. Performs a series of fits on the data. Use case: extract instantenous tau vs signal. Can be configured: # of exponentials in each fit, ROI tiume window, ROI of first fit, multiplier applied to the first ROI window (geometric progression along time axis)
- Curve TRPL: Added action, fit with spline. Can be configured: linear or semilogx, number of knots, minimum number of points between knots, maximum x span between knots.
- Curve JV: Supports the modified naming format of JV software. Handles the corresponding illumination spectrum and scan sweep directions.
- Curve plot type "bar" and "barh": added Curve Actions quick modifications: width/height, align, bottom/left. For keyword bottom/left, special keywords are implemented to refer to first, previous, next and last bar Curve.
- Curve plot type "bar" and "barh": added Curve Actions quick modifications: list of labels (xtickslabels or ytickslabels).
- Curve EQE, CurveCV, CurveCf, CurveTRPL, CurveMCA, CurveSIMS, GraphPLQY, GraphJscVoc: Added a mechanism to suggest graph axis labels, in case current labels appear not appropriate (e.g. following data transform).
- Script JV, can now process JV series where cells are identified by #1, #2 etc.

*Modifications*

- CurveMCA: the default scaling values offset and mult to compute energy from channel are now parsed from a text file, and not hard-coded anymore.
- CurveTRPL: Added option to restrict fit curve onto time range of interest ROI data. Slightly revised initial fit parameters.
- config.txt: revised default color palettes

*Other*

- Improved compatibility with numpy >=2.0. Please report remaining difficulties.
- Added logfile to locate possible bugs and errors: grapalog.log
- Cleanup of docstrings for the automatic documentation of grapa.
- Solved glitches with excess printed information
- Solved issue with graph to clipboard that prevented good operation of the feature
- Solved (hopefully) a bug with export and copy_to_clipboard, when alter keywords was specified.
- Reworked a number of internals, various refactoring. New python files curve_subclasses_utils.py, plot_curve.py, plot_graph.py, metadata.py
- Created tests, a bit for practical purpose but more to see how to make tests at all - but it is a start.





**Version 0.6.4.0**

*Released 07.11.2024*

*Additions*

- GUI: Added an option to modify the graph background color for visualization purpose. The output image remains with a transparent background, to be provided separately.
- GUI: Added a button to reverse the curve order. Ignores curve selection.
- Curve EQE: Bandgap derivative, added calculation of the bandgap PV by Rau (https://doi.org/10.1103/PhysRevApplied.7.044016), by averaging the energy weighted by EQE derivative value, over the derivative FWHM. Its value is slightly more sensitive to experimental noise than the derivative method. When the derivative peak is asymmetric, the value tends to be slightly higher than the derivative peak (up to 30meV ?).
- Curve EQE: Bandgap derivative, also added a fit to the derivative suited to best estimation of sigma value, intended for independent estimation of DeltaVoc_rad.
- Curve EQE: Added new function to calculate the short-circuit Voc loss, due to Jsc < Jsc_SQ. Calculation following Rau et al PRA 7 044016 (2017) DOI: https://doi.org/10.1103/PhysRevApplied.7.044016
- Curve EQE: Function ERE. Added the calculation Qe-LED by Rau (equivalent to ERE with geometrical factor fg=1), and the Voc bloss breackdown into DeltaVoc short-circuit, radiative, non-radiative. The center-of-mass of the PL peak (EQE*blackbody) is also provided for comparison purposes. The bandgap PV of Rau is used for calculations, or given by user. Note: changes in input bandgap is mostly accomodated in the DeltaVoc_rad value. Added auxiliary Curves to visualize data extraction fits. Added auxiliary Curve for parameter copy-paste.
- Curve Image: Can now configure levels using arange, linspace, logspace, geomspace. The parameter extend can also be set.
- Graph PLQY: implemented reading of output files of power dependency module.
- Script CVfT: From a set of C-f data acquired at different voltages and temperatures, provides C-V-f maps for each temperatures indicating low values for phase, as well as C-f, C-V with T and C-V with Hz plots.
- Script Cf: added Bode plot, |Z| and impedance angle versus log(frequency)
- Script Boxplot: Added summary graph presenting all generated boxplots

*Modifications*

- General: Conversion nm-eV is now calculated from Plank and c constants (previously: 1239.5, now about 1239.8419)
- General: graph colorize modifed behavior: if sameIfEmptyLabel, same colors are also applied in case label is hidden (labelhide)
- Curve EQE: Revised parametrization of bandgap by derivative peak method. The fit is now parametrized in unit of eV.
- Graph PLQY: when opening a file, added PLQY(time) as curve hidden by default
- Graph TinyTusker: various improvements
- Script Cf, image derivative: redesigned the image. The axes are now omega versus 1000/T (input data are in K, calulated on-the-fly with alter keyword). The fit curves of activation energies can be directly added onto the C-f derivative image.
- Script JV: Rp, Rs from acquisition software are now reported in summary files and in graphical summary (diode).
- Script JV: Rsquare fit quality restricted to the diode region is reported in the summary files and in graphical summary (diode). The marker size of the other fit parameters shrinks in case poor Rsquare values were obtained.
- Script Correlation: Improved detection of input parameters varied in a logarithmic manner.
- Script Correlation: Revised colorscale of plot "parseddata" for datasets with 2 input parameters

*Bug corrections*

- General: Solved a bug that prevented making figures with a unique subplot
- General: The property xtickslabels and ytickslabels can now be used also in conjunction with the property alter.
- General: Plot type fill_between and fill_betweenx now have more proper behavior.
- GUI: Small adjustments against MacOS dark mode
- GUI: Solved a bug that appeared when a tab was closed before the figure drawing was finished. Graphs drawn later on were not drawn correctly if contained several axes.

*Miscellaneous*

- General: Centralized physical constants in a unique file constants.py. Hopefully everything works as before.
- Implementation: new text files to store content of (now renamed) variables Graph.dataInfoKeysGraphData, Graph.graphInfoKeysData, Graph.headersKeys
- Implementation: tidy up the code at a number of places







**Version 0.6.3.1**

*Released 17.05.2024*

- GraphCf: parsing of Cf data more tolerant to variations in file format



**Version 0.6.3.0**

*Released 21.04.2024*

- New type of file supported, TinyTusker
- Can now open files directly from Openbis. This require the external library pybis, and access to the Empa-developped code openbis uploader 207.
- Added Curve action Autolabel: Curve labels can be generated based on a template, using the attributes values. Example: ${sample} ${cell}". Curve types: SIMS, Cf, CV, EQE, JV
- Boxplot: new possibility to add seaborn stripplot and swarmplot on top of boxplot
- Boxplot: GUI, added option to select the value of showfliers
- Boxplot: Added support for the whole list of parameters for ax.boxplot. Hopefully, not that many problems with unintentional keywords.
- Curve CV: added display and retrieval of carrier density and apparent depth, after using the function "Show doping at"
- Curve EQE: revised caculation of the bandgap by derivative method (choice of datapoints for fitting of maximum value)
- Curve JV: added data transform Sites' method, dV/dJ vs 1/(J-Jsc)
- Curve Subplot: added Curve Actions for easier modification of axis limits and labels
- Curve TRPL: revised parsing of acquisition time values. Should work with spectroscopic dataset as well.

*Bugs & code cleaning*

- Solved issue with leftovers on the canvas when changing figure e.g. size.
- Improved auto screen dpi. When opening a file, the graph should be displayed witih a size close to the maximum displayable value
- Refactored code for boxplot and violinplot
- Solved small bugs here and there, likely added a few new ones...




**Version 0.6.2.2**

*Released 12.10.2023*

*New data file format supported:*

- PLQY file Abt207
- GraphJV_Wavelabs: new file format to parse JV as well as MPP data files

*New features*

- Curve TRPL: new data processing function, Curve_differential_lifetime_vs_signal
- CurveSIMS: formatted for the Label, using python string template mechanisms and curve properties as variables. Maybe more useful than CurveSIMS..

*Bug corrections*

- CurveSIMS, bug recently introduced that prevented opening files under some conditions.





**Version 06.2.1**

*Released 11.09.2023*

*BUGS*
- Solved a bug in CurveJV that was preventing proper recognition of dark and illuminated curves in some cases, e.g. for scripts.




**Version 0.6.2.0**

*Released 30.08.2023*

- New file format: grapa can extract part of the data contained in a set of SCAPS software output data (.iv, .cv, .cf, .qe).
- New script: show correlation graphs between quantities taken from da tables, excel files. For Scaps simulation batches, shows the simulated curves as well as correlations between sweep variables and PV parameters.
- Curve Images: added conversion function, to convert matrix data into 3-column xyz format and conversely. The data is displayed in a new Graph.
- Curve Images: contour, contourf: revised the code for column/row extraction, transposition and rotation.
- GUI: Copy Image to clipboard now should preserve transparency
- GUI: "New empty Curve" nows inserts, and not append the new empty Curve.
- Curve TRPL: added function to calculate tau_effective, by 2 methods. A warning is issued if a tau value may risk to artifact the result.
- Curve TRPL: added functions to send fit parameters to clipboard. Also reports the weighted averages if no risk of artifact.
- Curve TRPL: fit should be more robust and have less convergence issues.
- Curve EQE: now parse the reference filename from the raw data file
- Curve JV: axvline and axhline are created with with thinner lines
- Curve JV: identification of sample and cell performed at opening; fields to edit available. Goal: identify challenging cases with new setup.
- Script JV: the "_allJV" now has axis labels

*BUGS*

- SIMS: solved a bug that was failing to perform quantification of SIMS relative yield. There was no indication that the normalization failed on the GUI, only in the command line window. As a result, curve ratios (e.g. GGI) ma have beed calculated in an erroneous manner
- A bug with Curve actions with widgets of type Checkbox. They values were always displayed as False.
- Script JV: returns a different graph (JVall). The other graphs could not be "saved" due to a technicality.
- Script JV: solved a bug that prevented parsing a datafile that was created after a first execution of the JV script which did not find that file.
- and a few minor bugs here and there




**Version 0.6.1.0**

*Released 19.04.2023*

*Features*

- GUI: it is now possible to open several files at once (finally!)
- Axis labels and graph title can now be entered as ['Quantity', 'symbol', 'unit'], with optional formatting additional dict element
- File format: grapa can now open JV files from Ossila sofware, rather primitive data parser.
- File format: grapa can now extract data from a certain .spe file format containing XRF data. The data parser is very primitive.
- Curve EQE current integration: added a checkbox to show the cumulative current sum.
- Curve Math: can now assemble a new Curve based on user-selected x and y data series of same length available within current Graph object.
- Curve JV: the code should now be insensitive to sorting of input data (extraction of parameters is done on a sorted copy of the data)
- Curve TRPL fit procedure: recondition the fitting problem, the fitting should be more robust and less prone to reaching max iterations
- Curve XRF MCA: retro-compatiblity of element labelling features
- Curve XRF: does not anymore overwrite spectral calibration if already set

*General*

- Ensured forward compatibility up to Winpython 3.10.40

*Bugs correction*

- Curve JV, can read date time.




**Version 0.6.0.0**

*Released 20.05.2022*

*Additions*

- Main GUI now handles several graphs at the same time, thanks to a tab mechanism. Hope this will be useful!
- Change in handling of escape sequences: \n, \t, etc. Should be compatible with some special characters with different charsets (e.g. alt+230 "µ" in both ascii and utf-8 file encoding) and latex commands with 2 backslashes (e.g. "\\alpha"). "\alpha" would fail due to the escaped "\a" special character, but "\gamma" should succeed). Possible loss of compatiblity with previous graphs, esp. with latex symbols - hence, new major version number.
- Axis limits: when axes limits cannot be computed with data transforms, the user input is used to set the axis limit. It is now possible to define axes limit values, when previously this could not be done. The default behavior remains that the user input for axis limits are transformed the same way as the plotted data.
- Popup Annotations: added a vertical scrollbar, changed the order of displayed elements
- scriptCV: The warnings are collected and reported at the end. Also, more robust processing of input files with slightly different formatting.
- CurveSpectrum: added correction function instrumental response 800nm
- CurveEQE: added an additional reference EQE spectrum (Empa PI 20.8%). The data loading mechanisms is modified and new reference spectra are now easy to add - see file datatypes/EQE_referenceSpectra.
- CurveXRF: added an annotation function to place labels for peak intensities according to the database https://xdb.lbl.gov/Section1/Table_1-3.pdf.
- CurveXRF: improved the loading of experiemntal parameters. The data are now stored inthe Curve where they belong
- CurveSIMS: add function to crop data inside ROI
- CurveTRPL: fitted curves can now be normalized with same parameters as the input data
- CurveArrhenius: added possibility to fit using a power law
- GUI: a major rework of the organisation of the GUI code. Possibilities to hide different panels. Little visible changes, but many possibilities for new bugs. Please let me know if you notice any!

*Bugs*
- Solved a certain number of those. Did not keep track.
- scriptJV: solved a bug with _JVall when processing several samples simultaneously
- Certainly added quite a few new bugs. Enjoy, my pleasure






Version 0.5.4.8
Released 24.03.2021
Modifications
- TRPL: the data type TRPL was  modified to properly load files generated using
  scripts.
- Modified loading of preference file. Hopefully a bit faster when opening many
  files.
- CurveJV: added data transform 'Log10 abs (raw)' (abs0), to plot the log of JV
  curve without Jsc subtraction
- scriptJV: also exports a file with a few statistical quantities of JV data
  (average, median, max)
- scriptJV: reduced amount of generated files. The individual summary images are
  not generated anymore as already plotted in combined format. Also, the
  different cells are exported as images (3x) and only once as text file.
- prepare future update of the GUI to support several graphs simultaneously
- a bit of code cleaning for better compliance with guidelines
BUGS:
- Mitigated bugs with recent versions of python (aka winpython version 3.9:
  issues with matplotlib 3.3.9; hidden curves in grey due to changes in tk)


Version 0.5.4.7
Modifications
- Adjusted width of GUI Curve action fields for CurveSIMS to prevent time-consuming redimensioning of the GUI interface.
- The Colorize GUI function can now print its effect in the console
Bugs
- Solved an issue with twiny ylim values


Version 0.5.4.6
Released 08.01.2021
Additions
- Added Integration function to the CurveTRPL and CurveSpectrum
Bugs
- Fixed a bug with Colorize function with curve selection where the order was not sorted under some conditions.



Version 0.5.4.5
Released 18.12.2020
Changes
- CurveTRPL: small changes in the handling of TRPL data label and normalization factors
- CurveCV: change in labeling for doping extracted at given voltage


Version 0.5.4.4
Released 18.11.2020
New features:
- Colorscale: it is now possible to colorize a selction of curves using colorscale, and not only the whole graph.
- Curve CV: a new function allows to display the doping at 0V (or other bias voltage value)
- Script CV: the automatic VC data processing now exports the doping at 0V on the plots N_CV versus depths. the oping at 0V as function of temperature is also reported in the NcvT summary.
- TRPL: Added a intensity normlization function, to compare intensities of time traces acquired with different instrument parameters (acquisition time, laser repetition frequency, time bin width). The normalized data are expressed in units of (cts+offset)/(repetition*duratio*binwidth) [cts/Hz/s/s]. It interplays with the existing offset feature, such that raw data can always be retrieved by setting the offset to 0 and removing the normalization.


Version 0.5.4.3
RELEASE DATE?
- ScriptJV: can now handle samples with cells identifiers with numeric > 9. The cell identifer is assumed to be with the form a1, b3, etc.
- CurveEQE: added a new analysis function, for crude estimate of thickness of layer with parasitic absorption (e.g. CdS). The deature can be hacked e.g. to reproduce absorption edge of perovskites.


Version 0.5.4.2
Released 13.08.2020
- Script JV: now also generates a compilation of (area-corrected) JV curves processed by the script
- Script JV: minor adjustments to the script.
- Script JV sample map: minor adjustments to the script, indication of columns not found.
BUGS
- SIMS: Solved an occasional bug with automatic edge detection that prevented opening SIMS data files.



Version 0.5.4.1
Released 11.05.2020
- Added method __len__() to the class Graph, returning the number of Curves.
- Added method __getitem__() to the class Graph, enabling call to Graph[0] or for c, curve in enumerate(graph).
- Added method __delitem__() to the class Graph, enabling to del Graph[0]. Calls Graph.deleteCurve(key)
- Added method attr() to the class Graph, a shorter alias to Graph.getAttribute()
- Added method attr() to the class Curve, a shorter alias to Graph.getAttribute()
Improvements
- Improved the reliability of CV and Cf script processing versus noisy data and incomplete input files.
BUGS
- The attribute label is now parsed as a string, so Curve labels such as "1", "2" can be used.


Version 0.5.4.0
NOT RELEASED
- Read CSV export files of the SquidAdmiral system
- Implemented automatic processing of Jsc-Voc data. Careful, the fit limits may be quite off from reasonable values.
BUGS
- Modified the output of scriptJV so samples with name purely numerical correctly proceed through the script
- Solved a bug in script Cf that prevented the correct display of the map dC/dln(f) vs T vs frequency.
- Solved a bug in script CV that stopped the execution when for some reason, negative values of N_CV were computed.



Version 0.5.3.3
Release 08.07.2019
Modifications
- The code was slightly modified to enable compatibility with winpython 3.6 (matplotlib 3.01)
- The data editor was revised and can now handle significanly larger datasets before speed becoms an issue (ca. 100'000 points instead of ~1'500)
Bugs
- CurveJV was modified to better fit JV curves of (mini-)modules. A warning is printed if input data may be provided in mV. Also, the area works as expected.




Version 0.5.3.2
Release 01.07.2019
Additions
- New values possible for keyword "alter": 'y' and 'x', with combination ['y','x'] enabling graph transposition
- Curve Cf: the dataset can now be displayed as "Frequency [Hz] vs Apparent depth [nm]"
- Script Cf: the "Frequency [Hz] vs Apparent depth [nm]" is now automatically generated
- The "axhline" and "axvline" keywords can be used to specify the formatting, independently for different sets of lines. Examples are provided.
Bugs
- CurveJV: when creating a CurveJV object, an "area" parameter is set by default with value 1.
- Script process JV: a cause for exception and script failure was corrected, in case of missing data.
- Script process JV: script execution should be more robust with untypical data. Basic JV parameters should be extracted, and failures with JV fits should not affect script execution. Missing Rs, Jo and ideality values might result.
- Prevents the data editor to crash when first Curve contains too many data.




Version 0.5.3.1
Release 20.08.2018
Additions
- A Nyquist plot is created when executing the script C-f.
- An image is created when executing the script C-f, showing the derivative as function of T and frequency.
- Shortcut were placed to tune the boxplot appearance: horizontal position, width of the boxes, presence of a notch, vertical or horizontal orientation.
- Keyboard shortcuts were added:
    Ctrl+h: hide (or show) the active curve
    Ctrl+Delete: delete the active curve
- References were added to the help function of CurveCf, papers from Walter, and Decock.
 Modifications
- EQE ERE estimate now discplays the input Voc of the device as a double check
- In the script treating C-V data, the units of the apparent doping density are now properly displayed in [cm-3] units and not [m-3]
- An additional example is provided for the keyword colorbar




Version 0.5.3.0
Release 20.06.2018
New features
- A data editor was implemented, accessible immediately below the graph.
- The Property edition tool was removed from the user interface. The 'New property' is now renamed 'Property editor'
Additions
- In the user interface, two buttons were added to reorder the stack and place the selected Curve at the top or at the bottom of the stack.
- CurveEQE: a new anaysis tool is provided, the external radiative efficiency. This estimates is computed from the EQE and the cell Voc.
- CurveJV: added a new data visualization: differential R, defined as dV/dJ.
- Added an option for the plot method 'fill'. Points with 0 value can be added at first and last data positions thanks to the keyword 'fill_padto0'.
- Added a value for keyword 'arbitraryfunction": [['grid', [True], {'axis': 'both'}]], which displays a grid at the major ticks locations
Modification
- CurveEQE: the default interpolation for the current calculation currentcalc is now 'linear' and not 'cubic'
- When trying to save a graph with special characters that cannot be saved in a text file, some clearer (and hopefully helpful) message is now displayed.
Bugs
- The stackplot method now properly ignores hidden Curves.
- The stackplot now hides the curves labels when the keyword labelhide is set
- Solved a bug for invalid input for the estimate of ERE cut wavelength
- Solved a bug with improper inputs for the subplots_adjust that frooze the graph



Version 0.5.2.3
Release 23.05.2018
Modifications:
- Jsc-Voc: the data separation Voc vs T now hides most created labels.
Bugs:
- Handling of colors in stackplot curves is now effective.
- Solved an issue wen saving files with fitted curves, which could not be opened when reloaded.
- in Jsc-Voc data treatment, solved a bug providing faulty default fitting range for J0 vs At.



Version 0.5.2.2
Release 23.04.2018
Bugs
- Bugs in some curve actions, where the curve was passed in argument.


Version 0.5.2.1
Released 13.04.2018
Bugs
- Bugs in rounding with infinity values, notably in EQE Curves


Version 0.5.2.0
Released 09.04.2018
Additions
- Actions specific can now be performed on several curves at the same time, provided The corresponding action is available on each selected curve. Example: bandgap from EQE curve, JV fit, fitted curve resample, etc.
- When extracting Voc(T) from Jsc-Voc data, the data can now be fitted to a certain range and the fit extrapolated to 0 with a single clic.
  Moreover the Voc @ T=0 are printed in the console.
- The determination of the optical bandgap from EQE curves can be restricted to a certain wavelength range, in the derivative method.
Modifications
- Curves created from curves actions (fit, etc) are now placed just after the selected curve.
- Improved the robustness of the JV curve fitting
- Adjusted precision of default parameters for TRPL fit, EQE exponential decay, and JscVoc curves.
- In the fits to TRPL data the tau are now non-negative, helping finding a good fit.
- SIMS data: the GGT keyword now refers to the ^72Ge+ trace and not ^70Ge+ anymore.
- The color picker popup now displays the current defined color, if possible.
Bugs
- Minor bug solved with overriding textxy values
- Bug solved that prevented the opening of the annotation popup with some input textxy values
- Legend location 'w' and 'e' were swapped
- Solved an issue that cause buttons to not disappear in the actions specific panel.
- Fit of JV curves, prevents creation of fit curves with non-sensical data in the 1e308 range
- Solved a bug in the output of summary file of boxplots, not correctly identifying the name of some sample names



Version 0.5.1.0
Release 18.03.2018
Additions
- CurveJV can now read an updated version of the TIV files
- In the Actions specific to the Curves, the quick access to the offset and muloffset attributes was changed to be Combobox instead of Entries.
- Added special keywords for offset and muloffset keywords: 'minmax' and '0max', which stretch the data from min to max, and 0 to max respectively.
- Added options to export Curves or Graph to clipboard with raw or screen data, and with or without properties.
- JscVoc curves: added a button to separate the data series as Voc vs T. The data is supposed to converge to the bandgap at T=0.
- CurveArrhenius: the fit range ROI is indicated in the attributes of the fitted Curve.
- CurveArrhenius: a new possibility is offered to define the x values after the curve creation.
Bugs
- Solved a graphical glitch in the annotation popup, regarding inappropriate "new" labels upon creation and deletion of annotations.
- Solved a glitch, the filename is not changed when copying the graph to the clipboard
Under the hood
- Moved the class GraphJV from the file curveJV to graphJV




Version 0.5.0.3
Small adjustments to make grapa installable via pip




Version 0.5.0.0
Release 11.02.2018
New major version number, indicating that grapa can installed using pip!
Otherwise no big changes in the software.
Additions:
- Can now read some reflectance files (Perkin Elmer spectrophotometer).
- CurveSpectrum has beed significantly extended, to offer some support to reflectance and transmittance curves. The following functions were added:
  - Correct for instrumental response, stored in datatypes/spectrumInstrumentalResponses.txt.
  - Compute the absorptance, defined as A = 1 - R - T
  - Estimate the absorption coefficient alpha, with input the thickness of the layer. A simple is also provided to account for the absorption in the substrate (see file datatypes/spectrumSubstrates.txt)
  The last 2 functions request selecting a transmittance/reflectance curve.
  More details are given in the manual.
Modifications:
- Added options to easily change the calibration of XRF files
Under the hood:
- Revised the code for opening spectrum files.



Version 0.4.3.3
Release 04.02.2018
Additions:
- CurveEQE: the calculation of cell Jsc now offers the choice of interpolation polynomial order, and the choice between AM1.5G and AM0 reference spectra.
Modifications
- CurveCV: solved a bug in the vertical axis label.
- Updated Manual
- Updated readme



Version 0.4.3.2
Release 27.01.2018
Additions
- Read support of some XPS csv files format
- A few options to spice the fill and fill_between plot methods
- Extended support for the ticks locator, in keyword 'arbitrayfunctions'. 2 examples are provided.
Modifications
- Clearer description of the keyword zorder
- Clearer description and examples for keyword arbitraryfunctions


Version 0.4.3.1
Release 22.01.2018
Additions
- Implemented a data parsing from file for Curve_Image. The X, Y coordinates can be read from first row, column (leave an empty tab in top-left corner)
- Added a new Button in the GUI to create an empty Curve, useful for subplots, insets, images
- Selecting a keyword in the property tree now changes the active keyword in "New property". If positive review the panel "Edit property" will be deleted.
- A few more examples to the linespec keyword
- CurveJV: throws a message when a duplicate is found in the V data series.
- New quick fields for capsize and ecolor (color of errorbars) for errorbar types of curves.
- Added Combobox entries in the annotation popup, to suggest possibilities to the user. Also revised some details in the user interface.
Modification
- Improved handling of invalid syntax in text, textxy and textargs keywords when loading files
- Restore good performance when zooming when the crosshair is not active
- Improved handling of unusual keywords
- Improved handling of ylim for CurveCV types with Mott-Schottky plotting.



Version 0.4.3.0
Release 05.01.2018
Major changes
- Modified prototype of Graph.plot() function. Now, by default the plot is saved only if a file name is provided, and the ifSubPlot preventing deletion of existing axes sets as True if an axis is provided in the figAx parameter.
Addition:
- Added 2 new types of plot: contour and contourf. These act similarly as imshow in the sens that they aggregate the next Curves with same x values.
- Better example for PL spectrum
- Added a button to set the graph axes limits to the values set by the zoom tool.
- The crosshair of the data picker now follows the mouse motion while the mouse button is pressed.
- Added actions to curve type image: aspect, interpolation, transpose, rotation
- Added Curve actions to help setting suitable parameters for curves types "errorbar" and "scatter"
- Modified Curves actions to include drop-down Combobox menus: Curve_SIMS, Curve_Spectrum, Curve_TRPL, Curve_Math, Curve_Image, Curve_Subplot
- xlabel and ylabel can now be entered as lists, under the form ['label', {'color':'b', 'size':6}]
- A new Curve keyword was added to the menu: zorder, which determines the drawing order. The legend order and the drawing order can then be tuned independently. The keyword was already functional previously.
- Improved file reading of TRPL files
- Offsets and multiplicative offsets cannow be entered as string fractions, ie. [0.8, '1/3e2']
- Added a checkbox to copy to clipboard the attributes of the Curves/Graph together with the data.
Modifications
- Modified prototype of Graph.plot() function, see above.
- Various minor improvements in the clarity of the GUI and of the text annotation popup
- Replaced text Entries by Drop-down menus whenever relevant in the GUI popup
- Revised fitting procedure for JV curves, by increasing the assumed noise level due to lamp fluctuations
- Replaced the default Entry fields with more user-friendly Combobox in some Curve actions.
- New property: replaced the Entry field by a Combobox, summarizing the provided examples or possible values
- Improved handling of font sizes in axis labels, titles and ticks labels
- Revised internal mechanism for insets (and to some extend subplots). The behavior should be more consistent and predictable, especially when no file are provided.
Bugs:
- Improved handling of \t in text and labels. Relevant notably for the $\tau$ greek letter etc.
- keyword labelhide works again



Version 0.4.2.1
Release 20.12.2017
Addition:
- Added 2 new types of plot: contour and contourf. These act similarly as imshow in the sens that they aggregate the next Curves with same x values.
Modifications
- Various minor improvements in the clarity of the GUI and of the text annotation popup
- Replaced text Entries by Drop-down menus whenever relevant in the GUI popup
Bugs:
- Improved handling of \t in text and labels. Relevant notably for the $\tau$ greek letter etc.



Version 0.4.2.0
Release 18.12.2017
Modifications
- The drop-down menus for Graph and Curves properties have been rearrange to provide clearer strucure.
- The default font size is now taken from the matplotlib rcParams
- Improved the handling by GUI of carriage returns (\n) in graph attributes (labels, title, etc)
- Added Help! buttons for Curves types subplots and insets.
- Removed most pointless possible actions for fitted EQE curves.
- Added a x axis transformation nm to cm-1 curve transform for curve type CurveSpectrum.
GUI
- Most actions of Curves can now be executed on several curves simultaneously, if several Curves are selected.
- Moved the button "Open all in folder" close to the Open and Merge buttons.
- Splitted in 2 the quick modifications fields for xlim and ylim
- The data picker can now create a textbox with the data cordinates.
- Revised the layout of the Frames which can be hidden in the GUI.
- The panel Template and COlorize are now hidden by default.
- The data picker is now hidden by default, in order to reduce the complexity of the interface.
- The Curve transform button, as well as the plot type button, are now drop-down lists, which activates as soon as the user select a value (no second button is necessary anymore)
Additions
- A checkbox was added just above the Update GUI button. If selected, most commands used to modify the graph will be transcripted in the console.
- A new keyboard shortcuts: Ctrl+Shift+c (copy curve data).
- The screen dpi is automatically reduced when a ligure larger than the canvas is opened. Alternatively the screen resolution is increased up to 100 dpi when a Graph sufficiently small is loaded.
BUGS:
- When applying a template, ignore default values of fontsize and subplot_adjust keywords
- The current graph is not lost when aborting a script procedure by clicking "cancel" in the folder choice dialog window.


Version 0.4.1.2
Release
Modifications
- Subplot -> subplots_adjust to customize subplot layout
Bugs
- Solved unnecessary warning message in text annotations popup, when no value was set to the legendproperty kwargs.
- Setting custom xticks labels will not override the xlim setting (same for axis)


Version 0.4.1.1
Released 01.12.2017
Additions
- The popup annotation manager has been extended to handle colors and fontsize. Also, when pressing <Return> the form is validated.
- The popup annotation manager also facilitates the modification of the legend and legend title. A few additional options are now possible - see these in the manager.
- Added a few keyboard shortcuts to the GUI: Ctrl+S (save), Ctrl+Shift+S (save as), Ctrl+O (open file), Ctrl+Shift+O (merge file), Ctrl+Shift+V (merge clipboard), Ctrl+Shift+N (new empty curve), Ctrl+R (refresh GUI). Pressing Return will validate most of the text entry fields.
- Allow a few buttons to be validated when the user presses "Return" in the corresponding text fields: quick modifs, edit property, new property, label and color selection of a Curve, actions on Curves.
- The graph title can now be customized, similarly as the legend title.
- New Curve type, Curve_Image (image) which handles images and matrix-organized data input. Plots the image indicated with keyword 'imagefile', otherwise greedily aggregates next curves with same x values and displays the aggragated matrix as an image. Based on matplotlib.axes.Axes.imshow()
- New subplots can also be created by using a Curve of type Curve_Subplot with empty or ' ' subplotfile attribute. The next curves are placed in that subplot, and the legends are shown at the correct location.
Modifications
- The keyword legendlocation has been renamed to legendproperties as to better reflect its use. Backward compatibility should be assured.
Bugs
- Improved imperfect behavior of the data picker.



Version 0.4.1.0
Released 13.11.2017
Additions
- A (hopefully) user-friendly window to manage the text annotations, accessible from the "Text annotations" button above the plot.
- XRF html file: CIGS composition also computed from Cu/Se, Ga/Se ratios
- Support of EQE files from CSU (Colorado State University, Sites' group)
- Several default image formats can be configured in the config.txt file, with a line like "save_imgformat	['.eps', '.png']"
- Curve EQE: The Urbach decay energy can now be fitted, as a Curve action.
- Curve SIMS: when generating a elemental ratio Curve, the user can now specify window and degree of a SAvistky-Golay smoothening filter to be applied on each of the elemental traces
- The boxplot script can now create boxplots from various data files and is not restricted to JV summaries. In each created graph the data are selected in the corresponding column in each file.

Support for subplots and insets. Examples can be found in examples/_subplots_insets.
- A suplot can be created by inserting or casting a Curve of type "subplot".
  When the attribute subplotfile is set and indicate a data file, that file is shown as a subplot to the graph. The created axis is then used for the next Curves to be plotted, until a new Subplot Curve is found.
- Possibility to place another Graph as an inset in a given graph.
  A Curve must be cast to type "inset". When the attribute insetfile is set and points to a data file, that file is shown as an inset to the graph. Else, the created axis is used for the next Curves.

Modifications:
- Updated the list of ratios keywords for the SIMS processing. Especially are available the ratio keywords cuznsn [Cu]/([Zn+]+[Sn]), cuge [Cu+]/[^70Ge+], and snge [Sn+]/[^70Ge+].
- Improved handling of units when creating boxplots. The units is now consistent with units user preferences.
- The boxplot and the JV maps scripts do now generate identical output format whether input files were generated at the acquisition machine or from grapa.
- Revised the way the global maps of the JV summaries are created.
- The boxplot script can now generate boxplots from any files a folder. If the initial file is not identified as a JV summary, a boxplot is created for each column of the data files.
- The boxplot script can now create boxplots from various data files and is not restricted to JV summaries. In each created graph the data are selected in the corresponding column in each file.
- Curves attributes are further tested for interpretation. Previous behavior is that unrecognized keywords were checked for their presence in the prototype of the used plot method (e.g. ax.plot, ax.semilgy, etc.).
  Now, in addition unused keywords are also checked for the presence of a setter method on the curves handle. Assuming a keyword key with value val and a (list of) handle created via h = ax.plot(...), the method h.set_"key"(val) will be called (provided it exists). Thus for example 'markevery': 3 can be correctly interpreted.

Bugs solved:
- The boxplots should be safe from a nan value in the data. The boxplot script crashed when such input was provided.
- The boxplot script was generating undesired files in the folder of the application launcher.
- The behavior of shifting Curves up or down is now consistent. The error messages are not displayed anymore.
- The software now correctly reads labels with including carriage returns when opening a file
- Corrected the red-to-green color of some graphs generated when mapping a JV summary file.





Version 0.4.0.2
Release 21.09.2017
Additions
- Implemented a configuration file. The file can be specified at startup in a command line launcher: ie. C:\pythonpath\python.exe C:\graphapath\grapa\GUI.py %1 config.txt
- The default presentation of graph axis units and symbols can be tuned in the config, using the keywords graph_labels_units ([], () or /) and graph_labels_symbols (True, False)
- The default colorscales can now be configured in the config file
- Export of files in .emf file possible. Inkscape is required to perform the conversion from an .svg image, and the path can be configured in the config file
- Export of statistics data (average, best, median) in a file while creating boxplots.
- Default image file format can now be configured in the config file.
- Export and reading of as .xml files are now supported.
- Improved example and support of font properties for legendtitle. Revised example for legendlocation.
- Improved pasrsing of acquisition parameters ehn opening EQE files.
Cosmetics
- The content of the last update in printed a few days after a new release, based on the versionNotes.txt file.
- Added small help close to the "Curve specific" drop-down menu in the data picker area, explaining the effect of the option. At the moment the checkbox has marked effect only for CurveCf.




Version 0.4.0.1
Release 07.09.2017
Additions
- Added SIMS files in the examples repository
- JV process now also exports a file with a summary of average, best cell and median of the 4 basic cell JV parameters
- Added a Checkbox to force separated x columns in saved files
- Renames example files to get "anonymous" examples




Version 0.4:
Release 07.07.2017
- The files have been reorganized in order to comply to the usual Python package organisation. In principle it should be possible to install the package etc.
  This involved a bit of file renaming, rewriting and so on. Existing scripts might not run immediately.
  The main changes are:
      - Files renamed (ie. remove the 'class_' prefix, set lowercase latters, etc.),
      - Class Measure was renamed into Graph.
      - Class xyCurve was renamed into Curve. Backward compatibility of saved graphs should be ensured.
Additions
- Added support for various matplotlib plot functions: acorr, angle_spectrum, bar, barbs, barh, cohere, csd, eventplot, fill_between, fill_betweenx, hexbin, hist, hist2d, magnitude_spectrum, phase_spectrum, pie, plot_date, psd, quiver, specgram, spy, stackplot, stem, step, triplot, xcorr
- Added a keyword "['key', value]" in the New property drop-down list. This allows creating arbitrary keywords, for example to provide the desired keyword to the plot methods.
- Added newCurve type: CurveMath, which enables basic arithmetic operations between Curves.
Changes
- Revised xticksstep, yticksstep so the values are more suitable (if the step is smaller than the default one, then the default values will be shown)
- Upgraded legendlocation keyword, which can handle kwargs to the legend() function cal. Example: {'loc': 'sw', 'bbox_to_anchor':(0.1, 0.05)}
- The legend font size can now be modified by performing update({'legendlocation':{'fontsize': 8}})
- The offste and muloffset can be directly edited in the Curves actions.
- Saving with the option "Transformed data (better not)" Transform now saves data modified for offset and muloffset
- Colorscales can now handle rbga quadruplets as well (and so is able to handle transparency)
- Clearer error messages when entering illegal colormaps parameters
- The hidden Curves are shown in grey color in the properties Treeview.
Corrections
- Script process JV: the cells can now be identified with capital letters (ie. B3 equivalent to b3)
- Script process JV: adjusted units (Voc [mV] instead of Voc [V], Rp and Rs in Ohmcm2)
Bugs:
- Corrected a bug preventing the opening of some SIMS profiles with complicated shapes of Ga+In curve
- Can now colorize correctly boxplots from JV summary files, with arbitrary RBG tri/quadruplets
- Corrected a bug preventing saving with the extension .TXT.
- Script process JV: the graphical summary files now contain the cell performance data. vminmax now can be set to correct values.
- Script JV sample map: can impose cmap (colorscale) from the JV summary file
Under the hood
- When scripting, can now perform arithmetic operation on curves (ex: curveA / curveB). By default will interpolate data on the merged x series.
- Curve.__add__ now accepts an argument 'operator', which can be set to 'sub', 'mult' or 'div'. Setting it to 'sub' overrides the sub argument.



Version 0.3.8.10
Release 11.05.2017
Additions:
- if 'subplots_adjust' last element is "abs", then the values given are intepreted as absolute margins instead of relative.
- Curve EQE: current calc now allows a ROI to be set.
- The Curve with type 'fill' now can get a legend
- Added a keyword 'colorbar' to Curves, which adds a colorbar to Curves with a cmap (i.e. scatter).
Corrections
- The color of scatter Curves in the legend is now shown properly
- The Colorize button with "repeat if no label" now only takes into account displayed Curves, and spans the entire colormap range.
- The JV boxplot script failed when mixing summaries of the JV acquisition software and generating when processing JV fits.


Version 0.3.8.9
Release 02.05.2017
Added:
- New type of plot: errorbar type of plot. 2 corresponding keywords were added: xerr, and yerr.
  Error bars can be specified for each datapoint, by setting the keyword 'type' of the next Curve(s) as 'errorbar_xerr' and/or 'errorbar_yerr'.
- New keyword: markeredgecolor
- When substracting Curves (i.e. Curve Spectrum dakr substraction), a keyword appears in the CUrve properties indicating which operation was performed.
Adjustments:
- The graph can now be shown larger than 600x400 px when resizing the application window.



version 0.3.8.8.
Release data 27.04.2017
Changes
- Added a "Copy image" button next to the Save buttons. This places an image of the graph in the system clipboard (Windows only)
- Added a Checkbox on template loading, allowing or not to modifying the Curve properties
- JV fit cell-by-cell and separately are more robust against unexpected data (bad JV curves, etc.)


Version 0.3.8.7
Release date 18.04.2017
Additions
- Can now read I-V data from heat-light soaking setup
- Direct options to quickly modify horizontal and vertical labels and limits (xlabel, ylabel, xlim, ylim)
- Added quick access of offset and muloffset Curve properties for Curves Spectrum, TRPL, MCA
- Added new Curve property 'vminmax'. Sets the min and max values for colormaps (cmap).
- Now can set a number as Curve label, which will be interpreted as a string.
- Curve Spectrum: a new keyword was added to the background substract function, taking or not into account the offset and the muloffset.
BUGS
- Solved an issue with cell area correction when batch-processing JV data, when only the dark or only the illuminated data where acquired.
- Solved a memory leak when running scripts JV, CV, Cf, boxplots. The plots are now closed after generation, freeing the memory.
- Solved a bug with mappings of JV summaries.
- xlim and ylim also work with '' to let the boundary adapt to the data, not only with inf and nan.
- Solved a bug preventing opening a file when a CurveEQE file had a numeric as label.



Version 0.3.8.6
Release date 05.04.2017
Additions
- Added a normalized semilogy plot plot type in the GUI, meant for TRPL curves.
Improvements
- Boxplot scripts: made it more robust regarding input files
BUGS
- Addition of Curves: removed duplicates with interpolate=1 option (notably accessible in Curve Spectrum)
- When selecting a new Curve type to cast in, can again select Curve Spectrum



Version 0.3.8.5
Release 03.04.2017
New features
- Curve Jsc-Voc: handles Jsc-Voc data pairs from the cryo setup. Functions are:
  Cell area normalization, N & J0 fitting, and data splitting into different temperatures.
  The plots A vs T, J0 vs T and J0 vs A*T are generated, and the last one can readily be fitted with an Arrhenius relation.
- Added a mode 'ExtrapolateTo0' to CurveArrhenius: it is actually not an Arrhenius relation, and does what it's name says it does (fit with 1st degree polynom)
  The Vbi vs T graph of the script Curve CV is of this type per default.
Adjustments
- Improved general software speed. (graph updates not initiated when moving in the property tree, updates are >2x faster)
- Button "Save" offers a choice in the filename if the software thinks it's at risk to erase raw data.
- Scatter curves can now be adjusted in color and size, provided the next curves have the keywords scatter_c or scatter_s. Offset and muloffset keywords can be used to adjust the size.
- TRPL: default fit ROI starts at 20 ns.
Bugs:
- Can read TIV files again correctly
Under the hood
- Adjustements to CurveArrhenius, it is now way easier to implement new models / data treatments.



Version 0.3.8.4
Release 30.03.2017
New features
- Added a "Substract dark" Button for Spectrum Curves.
  Choice is given to perform element-wise or by interpolation, as well as to update or replace the existing Curve.
  Any Curve can be casted into CurveSpectrum to perform a substraction, then casted back in his original type.
- Added a smoothening and binning function for TRPL Curves.
- Added a model of Arrhenius curves to interpret C-f data, with no weak temperature dependency.
- Added a Help! button to Spectrum, SIMS and EQE Curves.
Revisions
- Slightly revised sample maps. The colormap of the generated maps can be modified a posteriori using the keyword cmap, which accepts matplotlib keywords as well as list available in the Colorize option.
Bugs
- SIMS Curves: solved a bug which prevented to compute ratios with some Python versions.
- Prevent drawing vertical or horizontal lines at 0 value when the corresponding axis is in log scale (especialyl useful when opening JV files)
Background
- The operations Addition and Substraction can not be called on Curve objects, either element-wise or interpolated on x axis.


Version 0.3.8.3
Release 21.03.2017
Addition
- TRPL curves can be now be read smoothly.
  Can adjust vertical (background) and horizontal (time) offsets.
  Can fit exponential decay with abritrary number of exponentials, with or without fixed parameters.
  The fit residuals can also be shown.
Changes
- PL spectra of TRPL setup are opened as Curve Spectrum.
- Curve C-f datapoint picker is now hidden by default, as derivative makes little sense.
Bugs
- Curve C-f less prone to Exceptions when picking datapoints.



Version 0.3.8.2
Release 20.03.2017
Additions
- Curve CV: added a function, to fit Mott-Schottky plot where N_CV is minimum.
- Added automatic processing of folders of C-V files
- Added automatic processing of folders of C-f files
Modifications
- Output of J-V cell-by-cells processing is now is the same folder as the data.
Bugs
- Solved issues preventing proper cell-by-cell processing of J-V files.



Version 0.3.8.1
Released 13.03.2017
Some bug solving. Script for treatment of CV and Cf is under development.
Modifications
- Slightly reworked handling of semilog plots, can now also handle boxplot in semilogy graphs.
- The boxplot of J0 is restored, and now in semilogy scale.
Bugs solved
- Slightly updated the JV processing scripts, now is again doing what it was supposed to do.
  Solved issues with graphs ylim, apparent photocurrent.
- The crosshair curve selection falls back on the last curve if the previous selection was impossible.



Version 0.3.8
Released 06.03.2017
Completes and improves functionality added in intermediate 0.3.7.1 release. The following description repeats some previously mentionned new features.
Additions to Curves
- Added improved parsing of C-V and C-f data files
- Added a "Help!" button for Curve types CV, Cf, JV, Arrhenius.
- C-V curves transform: Mott-Schottky, N vs V, N vs depth.
- C-V curves functions: Fit on Mott-Schottky plot.
- C-f curves transform: -dC / d ln(f), C vs log(f).
- New Curve type: Arrhenius. Can fit Arrhenius relations from E vs T data, or omega vs T extracted from C-f data.
- New functionality: add a fixed offset to the data of HR2000 spectrometer
Additions to GUI
- Added a data picker. Works by clicking on the graph. Can be restricted to the nearest datapoint of a chosen Curve.
  The data picker can save the selected point. If selection was restricted to existing data, can either export as raw data, as visible on screen, or as Curve-dependent export (C-f Curve implements export of omega vs T instead of C vs f)
- New functionality: replace string in every labels
- Added a button, delete all hidden curves
Bugs solved
- Solved a bug preventing reading data from clipboard
- Now correctly loads EQE 20.4% cell and computes EQE current even if code is called from an external script
Internal organisation
- All definitions of Curves, interface with GUI, as well as file reading instructions are now outside the Measure and Curve classes.
  Additional Curve types and file reading instructions can be added without modifying existing files.




Version 0.3.7.1
Release 01.03.2017
Meant as intermediate release with only partial support of CV and Cf data.
Additions
- Added a data picker. Works by clicking on the graph. Can be restricted to the nearest datapoint of a chosen Curve.
- Added a button, delete all hidden curves
- Added a new functionality: replace string in every labels
- Added a "Help!" button for Curve types CV, Cf, JV.
Support for CV and Cf data
- Added improved parsing of C-V and C-f data files
- CV curves transform: Mott-Schottky, N vs V, N vs depth
- CV curves functions: Fit on Mott-Schottky plot.
- Cf curves transform: -dC / d ln(f)
Bugs
- Solved a bug preventing reading data from clipboard
- Now correctly loads EQE 20.4% cell even if code is called from an external script


Version 0.3.7
Release 26.01.2017
Additions:
- Added possibilities to colorize the curves of a graph, with some quick choices in the color schemes. Custom colormaps can be set and used.
- Added a quick fields for fast changes of Curve label and color
Adjustments
- Shows the current value when selecting a new property which is already set
- Shows current calibrated depth in SIMS profiles, if already set.
Various
- Improved compatibility for future versions of Python
- Corrected bug in JV batch processing (area correction not properly performed)



Version 0.3.6
Release 23.12.2016
- Added capability: EQE current
    (button available for EQE Curves)
- Improved default data processing of VJ curve series:
    - Cells parameters mapping revised
    - Added composite images of cells parameters mappings



Version 0.3.5.1
Release 15.12.2016
- Solved bugs in batch processing of JV files (script was not processing to the end)
- Solved bug when saving files, when first column was not the one with most datapoints.
    Some saved files since version 0.3.4 might be corrupted. Files can be opened in a spreadsheet and the data column structure can be restored manually.



Version 0.3.5
Release 01.12.2016
- Added support of SIMS data files with numbers separated by commas and not dots.
- Minor adjustements to the GUI.
- Solved bugs preventing to past data from clipboard to the current graph.



Version 0.3.4
Release 15.11.2016
- Added data import from clipboard.
- Added support for SIMS measurements. Supported features:
     - Edges recognition
     - Handling of composition adjustment based on ratios (explicit description of ratio, or keywords: ggi, cgi, cuzn, cuzn, znsn, sncuznsn)
     - Depth scaling.
- Rearranged graphic user interface
- Exported data are now slightly more compact, avoiding duplicates when subsequent Curves have identical x data.
- EQE curves are not not modified upon opening, only the display is scaled to [0,100].



Version 0.3.3
Release 28.10.2016
New features
- Added automatic creation of samples maps presenting JV parameters
- Added possibility to create scatter plot. The points colors is given by the value of the next curve.
- Added a new Curve type dedicated to XRF (MCA Curve), with Curve transform associated Channel <-> keV.
- Added support of Curves horizontal and vertical offset (constant) and muloffset (multiplier)
- Added support for multiple text annotations
- Added support or arbitrary arguments of the text annotate function. It is now possible to draw arrows on the graph.
- Added support of arbitrary functions of the figure main axes. Default example is the drawing of abitrary minor axis ticks and labels.
- Added a switch between linear and log-type plots. If set, the toggle overrides the 'type' curve attributes plot, semilogx,y, loglog.
General improvements
- The figure outer background in now white in the application.
- Improved detection of columns labels in generic (column-defined) files
- Legendlocation can also be set with se, nw typed keywords
- The selected curves now moves up/down together with the curve when changing the curves order
- When opening a graph te buttons at the toggles at the top also
- The Save button now does not work if the meastype keywork is set to something else than graph.
BUGS solved
- Exceptions during plots, semilogx, etc. are now catched. Argument debugging should be easier.
- Solved a bug when loading a plot with legend location.




Version 0.3.2
Release 11.10.2016
- CSV files: on opening now automatically identify the data separator (i.e. tab, space, comma, etc.)
- Added button Save, which automatically saves the data without asking for a filename.
- Add safety to stop "Save" action when opening an actual datafile. Based on the value of the measType attribute.
- Added button to duplicate (clone) a Curve
- Added button to copy the data of a curve to the system clipboard
- Allows xtickssteps and ytickssteps to be a list of ticks locations
- Font size also valid for graphs annotations
- JVCurve: add possibility to calculate Voc, Jsc, FF, Eff, MPP
- Bug: show again the move up/down buttons



Version 0.3.1
Release 22.09.2016
- Opening XRF HTML file now prints the composition according to calibration x=0.3
- Opening XRF MCA file: now suggest the user to open the html file to know the composition.
- JVCurve fit: Jl is now shown (but not functional). The value shown is the difference with Jsc.
- Allows xtickssteps and ytickssteps to be a list of ticks locations
- Automatically identify separators in csv files
Bugs solved:
- Save files from automated JV folder processing -> undesired ylim (inf, inf) is added
- Silenced error message when computing Voc of noisy JVCurve.
- EQE Curve: Empa20.4% is now also looked in a subfolder.
- Open file with fitted JV Curve: _popt parameter should be read correctly
- Improved JVCurve fit weights (better fit of some JV curves).
- When loading a template, do not add a label if none is provided


Version 0.3
Release 12.09.2016
- Added possibility to cast a Curve into another type
- Location of x and y labels now adjustable by user
- Position and size of the axis area can be modified using the subplots_adjust parameter
- Added support of secondaries axis: twinx, twiny, twinxy
- Graph spacings left, bottom, right, top adjustable by user
- Graph DPI can now be adjusted. Default output is now 300 dpi.
- JVCurve overloads the method update, to normalize the data if a new area is given. Thus manually changing the area now normalizes the data
- Default save folder is now that of the opened fil, or opened folder.
- Added bottons to shift curves up/down in the list
- Various bugs solved
   - Label now correctly loaded from saved data file
   - Computation of JV curve corrected from a mistake. Results are seemingly only marginally affected.
     NB: Jl is internally computed such that J(V=0) = Jsc.


Version 0.2.2
Release 01.09.2016
- JVCurve fit: added user-adjustable weight to the fit in the diode region behavior, for Curve fitting as well as for folder processing-
- Changing view (alter) should keep track of the set xlim and ylim parameters (axHline and axvline still do not).


Version 0.2.1
Release 30.08.2016
- Limited precision displayed when showing curve fit parameters (JV and EQE Curves)
- Added fit range for EQE Curves


Version 0.2
Release 29.08.2016
Modifications:
- Added system of template, to export and load graph and curves graphical settings.
- Improved functionality of 'Change View' button
- New checkbox: save original data, or modified according to 'Change View'
- New type of Curves: CurveEQE
   fit options: bandgap Tauc (EQE*eV)^2
- Curve type JVCurve: added Change view nm<->eV



Version 0.1
- First implementation of GUI
   Can open various data file, or merge with already opened data.
   Possibility to see and edit graph and curve properties
- Data and Graphs can be saved and loaded, in a human-readable format (1 .png image and the corresponding .txt data file)
- Data reading ability:
   Various types of measurements formats (JV, TIV, EQE, HR2000, XRF, MBE log)
   Some excel reading capabilities (database-like, or data in column-like)
- Action possible on folders: Graph all data file saparately or merged, process JV files separately or (dark/illum) grouped by cell
- New type of Curve: JVCurve
   Fit function: 1 diode &  resistor model
   Change view: Linear, Tauc
