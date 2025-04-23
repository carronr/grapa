import os
import sys
import glob
import pytest

from grapa import Graph, logger, logger_handler


@pytest.fixture
def grapa_folder():
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.realpath(os.path.join(path, "..", "grapa"))


class HiddenPrints:
    # to temporarily ignore grapa printouts during test session
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        logger.removeHandler(logger_handler)  # to disable prints (so also logger)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        logger.addHandler(logger_handler)  # need to update the logger as well


def open_files_in_subfolder(grapa_folder, subfolder, filenames, together=True):
    folder = os.path.join(grapa_folder, subfolder, filenames)
    # print("folder", folder)
    out = []
    for file in glob.glob(folder):
        assert not file.endswith(
            ".png"
        ), "there should not be an image in folder {}".format(folder)
        graph = Graph(file)
        assert len(graph) > 0, "expect file not empty"
        assert len(graph[0].x()) > 0, "expect data of first Curve not empty"
        out.append(graph)
    assert len(out) > 0, "expect at least one file to open in folder {}".format(folder)
    return out
