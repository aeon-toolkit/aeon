"""Test output suppression decorator."""

import sys

from aeon.testing.utils.output_suppression import suppress_output


@suppress_output()
def test_suppress_output():
    """Test suppress_output method with True inputs."""
    print(  # noqa: T201
        "Hello world! If this is visible suppress_output is not working!"
    )
    print(  # noqa: T201
        "Error! If this is visible suppress_output is not working!", file=sys.stderr
    )


@suppress_output(suppress_stdout=False, suppress_stderr=False)
def test_suppress_output_false():
    """Test suppress_output method with False inputs."""
    pass
