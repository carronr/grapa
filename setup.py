from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("readme.rst").read_text(encoding="utf-8")

setup(
    name="grapa",
    version="0.8.0.0",
    description="Grapa - graphing and photovoltaics analysis",
    author="Romain Carron",
    author_email="carron.romain@gmail.com",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "python-dateutil",
    ],
    packages=find_packages(include=["grapa", "grapa.*"]),
    package_data={
        "grapa": [
            "*.txt",
            "*.rst",
            "*.log",
            "howto/*.py",
            "howto/*.txt",
            "howto/*.pdf",
            "howto/*.odt",
            "howto/graphics/*.*",
            "datatypes/*.txt",
            "examples/*.*",
            "examples/_subplots_insets/*.*",
            "examples/boxplot/*.*",
            "examples/Cf/*.*",
            "examples/CV/*.*",
            "examples/EQE/*.*",
            "examples/HLsoaking/*.*",
            "examples/HLsoaking/52_Oct1143/*.*",
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
    long_description=README,
    long_description_content_type="text/x-rst",
)
