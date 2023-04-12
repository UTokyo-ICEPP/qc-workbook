"""
Capture stdout from C libraries into a file. This is especially useful for qbsolv.
Note: if some output are printed after the end of the program or messes with the
regular python code printing, try setting `PYTHONUNBUFFERED` to true:

     export PYTHONUNBUFFERED=1

"""

import os
import sys
from contextlib import contextmanager


@contextmanager
def stdout_redirect(to=os.devnull):
    '''
    Usage:

    .. code::

        import os

        with stdout_redirect(to=filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")

    `Source <https://stackoverflow.com/a/17954769>`_.
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    #### assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


@contextmanager
def capture_stdout(to=None):
    """
    Same as stdout_redirect, but if to is None the stdout content is first captured into a
    temporary file, then dumped to stdout. To avoid conflicts with stderr, the latter is flushed as well.
    :param to: filename, or none
    """
    tmpfile = None
    if to is None:
        import tempfile
        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        to = tmpfile.name

    with stdout_redirect(to=to):
        yield

    if tmpfile is not None:
        tmpfile.close()
        sys.stderr.flush()
        with open(to) as f:
            print(f"==> {to} <==")
            print(f.read())
