"""Utility exceptions and warnings for grapa
Concept:
The API generates custom exceptions and warnings.

- Exceptions are what prevents a graph to be drawn or saved. Plotting is interrupted.
  Custom classes based on GrapaError for later filtering

- Lesser issues e.g. incorrect input in a plot property result in warnings
  Custom category based on GrapaWarning for later filtering

- A log file (grapalog.log) registers exceptions (as errors) and warnings

- (at the moment) the logger also prints the message in the standard output.
  (in the future, should not do that, grapa API should not print anything, only its GUI)


The GUI

- (at the moment) Intercept exceptions and warnings, ignores GrapaError and GrapaWarning

- (in the future, should print the messages of GrapaError and GrapaWarning)


@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import warnings


# WRNINGS
class GrapaWarning(Warning):
    """Base class for custom warnings. Its purpose is to be filtered out by GUI"""


# EXCEPTIONS
class GrapaError(Exception):
    """Base class for grapa Exception.
    By default, ignored by GUI as supposedly handled at lower level."""

    def __init__(self, *args, report_in_gui=False):
        super().__init__(*args)
        self.report_in_gui = report_in_gui


class FileNotCreatedError(GrapaError):
    """Raise when a file could not be created"""


class FileNotReadError(GrapaError):
    """Raise when could not read a file"""


class IncorrectInputError(GrapaError):
    """Raise when e.g. ehenever matplotlib is not happy with user input"""


# no raise_error, because want to keep pylance analysis 'code unreachable'
# -> directly in relevant code
def issue_warning(
    logger,
    msg: str,
    *args,
    category=GrapaWarning,
    severity_log="warning",
    exc_info=False
):
    """Logs a Warning in the logger, and issue a warnings.warn"""
    # for grapa maintainer
    if logger is not None:
        if severity_log == "error":
            logger.error(msg, *args, exc_info=exc_info)
        else:
            logger.warning(msg, *args, exc_info=exc_info)
            if severity_log != "warning":
                print("issue_warning issue a warning,", severity_log, "not implemented")
    # for the external user
    warnings.warn(msg % args, category=category)
