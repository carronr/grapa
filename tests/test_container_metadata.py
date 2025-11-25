"""
Tests for the content of module metadata.py
"""
import pytest

from tests import HiddenPrints  # grapa_folder
from grapa import Curve
from grapa.utils.container_metadata import MetadataContainer


@pytest.fixture
def container():
    """Returns a container"""
    curve = Curve([[1,2] ,[3,4]], {})
    out = MetadataContainer(curve)
    out.update({"a": 1, "b": 2})
    yield out
    # after yield: code for teardown


def test_container_metadata_methods1(container):
    """Test class ContainerMetadata"""
    # init
    assert len(container) == 2
    # get
    assert container.get("a") == 1
    assert container.get("b") == 2
    assert container.get("c") == container.VALUE_DEFAULT
    assert container.get("A") == 1
    assert container.get("A", -1) == 1
    assert container.get("D", -1) == -1
    # values
    assert container.values() == {"a": 1, "b": 2}
    assert container.values(["a"]) == {"a": 1}
    assert container.values(["b", "a"]) == {"b": 2, "a": 1}
    # has_attr
    assert container.has_attr("a")
    assert container.has_attr("B")
    assert not container.has_attr("c")
    # is_attr_value_default
    assert MetadataContainer.is_attr_value_default("")
    assert not MetadataContainer.is_attr_value_default(None)
    assert not MetadataContainer.is_attr_value_default(1)
    assert not MetadataContainer.is_attr_value_default("a")
    # update
    container.update({"B": 3})
    assert container.get("b") == 3
    container.update({"d": 4})
    assert container.get("D") == 4
    # items
    expected = {"a": 1, "b": 3, "d": 4}
    for key, value in container.items():
        assert container.get(key) == value == expected[key]
    assert len(container.items()) == len(container.values())


def test_container_metadata_methods2(container):
    """further tests"""
    # pop
    assert container.pop("b") == 2
    container.update({"B": 10})
    assert container.pop("b") == 10
    assert len(container) == 1
    # clear
    container.clear()
    assert len(container) == 0


def test_container_metadata_special(container):
    """tests centered on special methods __something__"""
    # str
    with HiddenPrints():
        print(container)
    # __call__, __getitem__
    assert container("A") == container.get("a") == container["a"] == 1
    assert container("b") == container.get("B") == container["B"] == 2
    assert container("c") == container.get("c") == container["C"] == ""
    assert "a" in container
    assert "B" in container
    assert "c" not in container
    # default value; __len__
    assert container("d", 1) == container.get("D", 1) == 1
    assert container("c", None) is None
    with pytest.raises(Exception):
        container[5] = 5  # must fail, keys must be str
    # __delitem__
    container["e"] = 5
    assert len(container) == 3
    del container["e"]
    assert len(container) == 2
    # __iter__
    assert len([c for c in container]) == 2
    assert [c for c in container] == ["a", "b"]
    # __setitem__
    container["A"] = 10
    assert container("A") == container.get("a") == container["A"] == 10
    container["a"] = "abc"
    assert container("A") == container.get("a") == container["A"] == "abc"
    # clear
    container.clear()
    assert len(container) == 0
