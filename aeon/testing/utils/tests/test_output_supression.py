"""Test output suppression decorator."""

import io
import sys

from aeon.testing.utils.output_suppression import suppress_output


def test_suppress_output():
    """Test suppress_output method with True inputs."""

    @suppress_output()
    def inner_test():

        print(  # noqa: T201
            "Hello world! If this is visible suppress_output is not working!"
        )
        print(  # noqa: T201
            "Error! If this is visible suppress_output is not working!", file=sys.stderr
        )

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    inner_test()

    assert stdout_capture.getvalue() == "", "stdout was not suppressed!"
    assert stderr_capture.getvalue() == "", "stderr was not suppressed!"


def test_suppress_output_false():
    """Test suppress_output method with False inputs."""

    @suppress_output(suppress_stdout=False, suppress_stderr=False)
    def inner_test():
        print("This should be visible.")  # noqa: T201
        print(  # noqa: T201
            "This error message should also be visible.", file=sys.stderr
        )

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    inner_test()

    assert (  # noqa: T201
        "This should be visible." in stdout_capture.getvalue()
    ), "stdout was incorrectly suppressed!"
    assert (  # noqa: T201
        "This error message should also be visible." in stderr_capture.getvalue()
    ), "stderr was incorrectly suppressed!"
