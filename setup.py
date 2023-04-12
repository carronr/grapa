from setuptools import setup

setup(
    name='grapa',
    version='0.6.1.0',
    description='Grapa - graphing and photovoltaics analysis',
    author='Romain Carron',
    author_email='carron.romain@gmail.com',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
    packages=[
        'grapa',
        'grapa.datatypes',
        'grapa.scripts',
        'grapa.gui',
    ],
    package_data={'grapa':[
        '*.txt',
        'manual/*.py',
        'manual/*.txt',
        'manual/*.pdf',
        'datatypes/*.txt',
        'examples/*.*',
        'examples/boxplot/*.*',
        'examples/Cf/*.*',
        'examples/CV/*.*',
        'examples/EQE/*.*',
        'examples/JscVoc/*.*',
        'examples/JV/SAMPLE_A/*.*',
        'examples/JV/SAMPLE_B_3layerMo/*.*',
        'examples/JV/SAMPLE_B_5LayerMo/*.*',
        'examples/SIMS/*.*',
        'examples/Spectra/*.*',
        'examples/TIV/dark/*.*',
        'examples/TIV/illum/*.*',
        'examples/TRPL/*.*',
        'examples/XPS/*.*',
        'examples/XRF/*.*',
    ]},
    license='MIT',
    url='https://github.com/carronr/grapa/',
	long_description='Grapa is a python package providing a graphical interface and the underlying code dedicated to the visualization, analysis and presentation of scientific data, with a focus on photovoltaic research.',
)
