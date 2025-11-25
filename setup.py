from setuptools import setup

setup(
    name="grapa",
    version="0.7.1.0",
    description="Grapa - graphing and photovoltaics analysis",
    author="Romain Carron",
    author_email="carron.romain@gmail.com",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "python-dateutil",
    ],
    packages=[
        "grapa",
        "grapa.datatypes",
        "grapa.gui",
        "grapa.scripts",
        "grapa.utils",
        # "tests",
    ],
    package_data={
        "grapa": [
            "*.txt",
            "*.rst",
            "*.log",
            "manual/*.py",
            "manual/*.txt",
            "manual/*.pdf",
            "datatypes/*.txt",
            "examples/*.*",
            "examples/_subplots_insets/*.*",
            "examples/boxplot/*.*",
            "examples/Cf/*.*",
            "examples/CV/*.*",
            "examples/EQE/*.*",
            "examples/JscVoc/*.*",
            "examples/JV/mix/*.*",
            "examples/JV/SAMPLE_A/*.*",
            "examples/JV/SAMPLE_B_3layerMo/*.*",
            "examples/JV/SAMPLE_B_5LayerMo/*.*",
            "examples/JV/SAMPLE_C/*.*",
            "examples/PLQY/*.*",
            "examples/SIMS/*.*",
            "examples/Spectra/*.*",
            "examples/TIV/dark/*.*",
            "examples/TIV/illum/*.*",
            "examples/TRPL/*.*",
            "examples/XPS/*.*",
            "examples/XRF/*.*",
        ]
    },
    license="MIT",
    url="https://github.com/carronr/grapa/",
    long_description="Grapa is a python package providing a graphical interface and "
                     "the underlying code dedicated to the visualization, analysis "
                     "and presentation of scientific data, with a focus on "
                     "photovoltaic research.",
    long_description_content_type="text/plain",  # "text/markdown",  # ADD THIS LINE
)
