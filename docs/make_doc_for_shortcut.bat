SETLOCAL
: SET "PATH=C:\_python\_python\WPy64-31160\python-3.11.6.amd64\Scripts\;%PATH%"

SET "SPHINXBUILD=C:\_python\_python\WPy64-31160\python-3.11.6.amd64\Scripts\sphinx-build.exe"

rmdir %cd%\source\_autosummary\ /s /q

CALL make clean
CALL make html source build

PAUSE 