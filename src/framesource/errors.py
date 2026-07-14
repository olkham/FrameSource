"""Exception hierarchy for FrameSource.

All library-specific errors derive from :class:`FrameSourceError` so callers
that want to catch "anything FrameSource raises" have a single type to
catch. At the same time, several of these exceptions are *also* subclassed
from the built-in exception type that FrameSource historically raised at the
same call site (e.g. ``ImportError`` for missing optional dependencies,
``ValueError`` for bad/unknown arguments).

This dual inheritance is deliberate: it preserves backwards compatibility.
Existing user code written against earlier FrameSource versions may contain
``except ImportError:`` or ``except ValueError:`` clauses around calls into
this library. Because ``MissingDependencyError`` is-a ``ImportError`` and
``UnknownSourceTypeError`` is-a ``ValueError``, that old code keeps working
unchanged, while new code can catch the more specific FrameSource types (or
the common ``FrameSourceError`` base) for finer-grained handling.
"""

from typing import Optional


class FrameSourceError(RuntimeError):
    """Base class for all errors raised by FrameSource.

    Catch this to handle any FrameSource-specific failure without needing
    to know which built-in exception type it also derives from.
    """


class MissingDependencyError(FrameSourceError, ImportError):
    """Raised when an optional third-party dependency is not installed.

    Inherits from :class:`ImportError` so existing ``except ImportError:``
    handlers written against earlier FrameSource versions continue to catch
    this exception unchanged.

    Attributes:
        package (str): Name of the missing package (or packages).
        extra (Optional[str]): Name of the FrameSource extra that installs
            the dependency (e.g. ``'audio'`` for ``pip install
            framesource[audio]``), if applicable.
    """

    def __init__(self, package: str, extra: Optional[str] = None, details: Optional[str] = None):
        """Initialize the error.

        Args:
            package: Name of the missing package (or packages).
            extra: Name of the FrameSource extra that installs the
                dependency, if any. When provided, the message includes an
                install hint (``pip install framesource[<extra>]``).
            details: Additional details to append to the message, such as
                the original ``ImportError`` text.
        """
        self.package = package
        self.extra = extra

        message = f"{package} is required but not installed."
        if extra:
            message += f" Install with: pip install framesource[{extra}]"
        if details:
            message += f" ({details})"

        super().__init__(message)


class UnknownSourceTypeError(FrameSourceError, ValueError):
    """Raised when a requested capture source type is not registered.

    Inherits from :class:`ValueError` so existing ``except ValueError:``
    handlers written against earlier FrameSource versions continue to catch
    this exception unchanged.
    """


class NotConnectedError(FrameSourceError):
    """Raised when an operation requires a connected source but it isn't.

    Reserved for use by subclasses and downstream code that wants to signal
    reads/operations attempted on a disconnected capture source. FrameSource
    itself does not currently raise this from its read paths.
    """
