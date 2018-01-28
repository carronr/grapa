CURRENT CAPABILITIES


General features
- Open and merge a wide range of experimental data. Data loading possible from clipboard.
- Graphical data presentation and graph customisation.
- Graph export in various file formats with user-defined dpi. Additionally create a .txt file with the raw data for later modifications.
  The supported image formats are the one supported by the savefig function of matplotlib. .emf is also supported, provided a path is set to a inkscape executable in the config.txt file.
- Single clic switch between data-relevant views (e.g. log-view for JV data)
- 2-clics switch between linear or logarithmic data visualisation.
- A data picker is implemented, to select and store chosen data points.
- Can colorize a graph using a few color gradients.
- Some data types can be fitted. The parameters of the fitted curves can be modified a posteriori.
- Data can be loaded from other files or from clipboard, then converted to the desired data type to benefit from data visualisation and processing and fitting options.



Push-button automated scripts:
- Load all files from one folder
- JV data fit cell-by-cell:
	asks for a folder,
	loads all JV data in that folder,
	perform area correction if 2-column cell/area file is found,
	select for each cell one dark and one illuminated, if any,
	fit the data with the single diode with 2 resistors model,
	output compiled dark+illuminated graphs, a compiled database of the cell properties, and graphical summaries.
- JV data fit separately: fit all JV files in a folder and compiles the data in a database file.
- JV sample maps:
	asks for a JV database file (either output of JV setup, or of cell-by-cell script above)
	output a graphical representation of each cells JV parameters
- JV boxplots:
	asks for a folder, open each JV databse files and create boxplots for each relevant JV parameter.
- Processing of C-f data series
- Processing of C-V data series


Supported formats:
- EQE files
	Data visualisation: EQE vs eV, Tauc plot (E*EQE)^2 vs E
	Functions:
		Bandgap from Tauc (E * EQE)^2, manual tuning of the curve fit possible
		Bandgap from derivative
		Bandgap from exact Tauc method (E * ln(1-EQE))^2
		EQE current, derived from the CurrentCalc Matlab software
		Loading the Empa 20.4% cell EQE
- JV files
	Data visualisation: logarithmic view log(abs(J - Jsc))
	Functions:
		Area correction
		Calculation of Voc, Jsc, FF, Eff, MPP
		Fit of the JV curve using diode with 2 resistors modeL
		Manual tuning of the curve fit possible, as well as resampling to higher number of points.
- TIV files: idem as JV files
- XRF MCA
	Data visualisation: Conversion channel <-> Energy [keV]
- XRF html files
	Quick access to CIGS fit peak areas.
	Computes composition of CIGS layers
- HR2000 USB spectrometer
	Data visualisation: switch from nm to eV
	Functions:
		Add arbitrary offset (although a dark is more suited!)
		Convert x axis data from nm to eV
- PL spectra
	See HR2000 spectra
- SIMS files
	Reads both dots- and commas-separated data.
	Data visualisation: Sputter time or depth (once calibrated)
	Functions:
		Some curves are automatically hidden at data loading.
		Edge detection on any elemental trace
		Depth calibration
		Adjustement of elemental yield coeficients to match known elemental ratios over some data range
		Computation of arbitrary elemental ratios
		Generation of arbitrary elemental ratio curves (e.g. GGI, Ga+In, etc.)
- C-V files
	Data visualisation: Mott-Schottky plot, Carrier density N vs V, N vs depth
	Functions: Fit Mott-Schottky plot.
- C-f files
	Data visualisation: semilogx, derivative
- TRPL decays
	Data visualisation: semilogy
	Functions:
		Add temporal offset, add vertical (background level) offset
		Fit decay with constant plus arbitray number of exponentials functions, with possibility of fixing some parameters. The residuals can also be shown.
- CSV files
- Generally, a wide range of column-organised data files (txt or excel files)
