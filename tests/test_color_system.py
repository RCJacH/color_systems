from color_systems.version import pkg_name, pkg_version


def test_call_color_system_version():
    """Checks that module is registered and visible in the meta data."""
    assert pkg_name
    assert pkg_version
