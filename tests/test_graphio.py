import pytest

from . import HiddenPrints
from grapa.utils.graphIO import _identify_delimiter


def test_identify_delimiter():
    def prepare(separator):
        lines_base = ["abc\tdef", "0\t1", "2\t3"]
        return [line.replace("\t", separator) for line in lines_base]

    # successes
    for delim in ["\t", ",", ";", " "]:
        with HiddenPrints():
            identified = _identify_delimiter(prepare(delim))
        assert identified == delim
    # failures
    with HiddenPrints():
        identified = _identify_delimiter(prepare("\n"))
    assert identified == "\t"


@pytest.mark.skip("Not yet implemented")
def test_export():
    raise NotImplementedError


@pytest.mark.skip("Not yet implemented")
def test_export_xml():
    raise NotImplementedError
