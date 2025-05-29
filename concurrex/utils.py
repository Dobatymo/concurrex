import logging
import threading
from concurrent.futures._base import FINISHED, Future
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from genutility.callbacks import Progress as ProgressT
from genutility.time import MeasureTime
from typing_extensions import ParamSpec, Self

if TYPE_CHECKING:
    import numpy as np

T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")

logger = logging.getLogger(__name__)


class _Unset:
    pass


WAIT_TIMEOUT = 10.0

JOIN_TIMEOUT = 10.0


def get_extra(self: Any) -> Dict[str, str]:
    return {"object": f"{type(self).__name__}-{id(self):x}"}


def wait_all(
    objs: Sequence[Union[threading.Event, threading.Condition]],
    total_timeout: Optional[float] = None,
) -> bool:
    if total_timeout is None:
        for obj in objs:
            obj.wait()
    else:
        with MeasureTime() as dt:
            for obj in objs:
                effective_timeout = max(0.0, total_timeout - dt.get())
                if not obj.wait(effective_timeout):
                    return False

    return True


def _debug_wait(
    objs: Sequence[Union[threading.Event, threading.Condition]],
    total_timeout: Optional[float] = None,
    per_wait_timeout: float = WAIT_TIMEOUT,
    extra: Optional[dict] = None,
    stacklevel: int = 2,
) -> bool:
    """
    Waits for a sequence of threading.Event or threading.Condition objects to be signaled.

    Each object is waited on at least once. If not all objects are signaled after a pass,
    the function checks whether the total timeout has been exceeded and exits early if so.
    Warnings are logged for each object that is not yet signaled, and when the total timeout
    is exceeded.

    Note: Conditions must be waited on with their lock held. This function does not enforce
    proper Condition usage but will emit a warning if used without acquiring the lock.

    Parameters:
        objs: A sequence of Event or Condition objects to wait on.
        total_timeout: Optional maximum total time in seconds to wait for all objects.
        per_wait_timeout: Maximum time in seconds to wait per object per loop iteration.
        extra: Optional dict passed to the logger for additional context.
        stacklevel: passed to the logger

    Returns:
        True if all objects were signaled within the total timeout, False otherwise.
    """

    remaining_objs = list(objs)
    with MeasureTime() as dt:
        for obj in remaining_objs:
            if isinstance(obj, threading.Condition):
                try:
                    if not obj._is_owned():
                        logger.warning("Waiting on %s without owning the lock", obj, extra=extra, stacklevel=stacklevel)
                except AttributeError:
                    pass  # _is_owned may not be available in all Python versions

        while remaining_objs:
            objs_waiting: List[Union[threading.Event, threading.Condition]] = []
            for obj in remaining_objs:
                if total_timeout is not None:
                    effective_timeout = min(per_wait_timeout, max(0.0, total_timeout - dt.get()))
                else:
                    effective_timeout = per_wait_timeout

                if not obj.wait(effective_timeout):
                    logger.warning(
                        "%s not signaled after %.02f seconds", obj, dt.get(), extra=extra, stacklevel=stacklevel
                    )
                    objs_waiting.append(obj)

            remaining_objs = objs_waiting

            if remaining_objs and total_timeout is not None and dt.get() >= total_timeout:
                logger.warning(
                    "Total timeout of %.02f seconds exceeded before all objects were signaled",
                    total_timeout,
                    extra=extra,
                    stacklevel=stacklevel,
                )
                return False

        return True


def debug_wait(
    objs: Sequence[Union[threading.Event, threading.Condition]],
    total_timeout: Optional[float] = None,
    per_wait_timeout: float = WAIT_TIMEOUT,
    extra: Optional[dict] = None,
) -> bool:
    if __debug__:
        return _debug_wait(objs, total_timeout, per_wait_timeout, extra, stacklevel=3)
    else:
        return wait_all(objs, total_timeout)


def debug_join(
    threads: Sequence[threading.Thread],
    extra: Optional[dict] = None,
    timeout: float = JOIN_TIMEOUT,
    stacklevel: int = 2,
) -> None:
    with MeasureTime() as dt:
        while threads:
            threads_alive: List[threading.Thread] = []
            for thread in threads:
                thread.join(timeout)

                if thread.is_alive():
                    logger.warning(
                        "Thread %s still alive after waiting for %.02f seconds",
                        thread.name,
                        dt.get(),
                        extra=extra,
                        stacklevel=stacklevel,
                    )
                    threads_alive.append(thread)
            threads = threads_alive


class Result(Generic[T]):
    __slots__ = ("result", "exception")

    def __init__(
        self,
        result: Union[Type[_Unset], T] = _Unset,
        exception: Optional[BaseException] = None,
    ) -> None:
        if (result is _Unset) == (exception is None):
            raise ValueError("Either result or exception must be given")

        self.result = result
        self.exception = exception

    def __eq__(self, other) -> bool:
        return (self.result, self.exception) == (other.result, other.exception)

    def __lt__(self, other) -> bool:
        return (self.result, self.exception) < (other.result, other.exception)

    def __hash__(self) -> int:
        return hash((self.result, self.exception))

    def get(self) -> T:
        if self.exception is not None:
            raise self.exception
        assert self.result is not _Unset  # for mypy
        return self.result

    def __str__(self) -> str:
        if self.exception is not None:
            return str(self.exception)
        return str(self.result)

    def __repr__(self) -> str:
        if self.exception is not None:
            return repr(self.exception)
        return repr(self.result)

    @classmethod
    def from_finished_future(cls, f: "Future[T]") -> "Result[T]":
        if f._state != FINISHED:
            raise RuntimeError(f"The future is not yet finished: {f._state}")

        return cls(f._result, f._exception)

    @classmethod
    def from_future(cls, f: "Future[T]") -> "Result[T]":
        try:
            return cls(result=f.result())
        except Exception:
            return cls(exception=f._exception)

    @classmethod
    def from_func(cls, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> "Result[T]":
        try:
            return cls(result=func(*args, **kwargs))
        except Exception as e:
            return cls(exception=e)


class CvWindow:
    def __init__(self, name: Optional[str] = None) -> None:
        import cv2

        self.name = name or str(id(self))
        self.cv2 = cv2
        self.cleanup = False

    def show(self, image: "np.ndarray", title: Optional[str] = None) -> None:
        self.cleanup = True
        self.cv2.imshow(self.name, image)
        if title is not None:
            self.cv2.setWindowTitle(self.name, title)
        self.cv2.waitKey(1)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        if self.cleanup:
            self.cv2.destroyWindow(self.name)


class NumArrayPython(Generic[T]):
    def __init__(self, *args: T):
        self._arr = list(args)
        self._lock = threading.Lock()

    def __iadd__(self, other: "NumArrayPython") -> Self:
        with self._lock:
            for i in range(len(self._arr)):
                self._arr[i] += other._arr[i]
        return self

    def __isub__(self, other: "NumArrayPython") -> Self:
        with self._lock:
            for i in range(len(self._arr)):
                self._arr[i] -= other._arr[i]
        return self

    def __len__(self) -> int:
        return len(self._arr)

    def to_tuple(self) -> Tuple[T, ...]:
        with self._lock:
            return tuple(self._arr)

    def __iter__(self):
        return iter(self._arr)


class NumArrayAtomicsInt:
    def __init__(self, a: int, b: int, c: int) -> None:
        self.val = a * 2**32 + b * 2**16 + c


class NumArrayAtomics:
    def __init__(self, a: int, b: int, c: int) -> None:
        import atomics

        self.a = atomics.atomic(width=16, atype=atomics.INT)
        self.a.store(a * 2**32 + b * 2**16 + c)

    def __len__(self) -> int:
        return 3

    def __iadd__(self, other: NumArrayAtomicsInt) -> Self:
        self.a.fetch_add(other.val)
        return self

    def __isub__(self, other: NumArrayAtomicsInt) -> Self:
        self.a.fetch_sub(other.val)
        return self

    def to_tuple(self) -> Tuple[int, ...]:
        return tuple(self)

    def __iter__(self) -> Iterator[int]:
        rem, c = divmod(self.a.load(), 2**16)
        a, b = divmod(rem, 2**16)
        return iter([a, b, c])


def with_progress(
    _func: Callable[[Callable, Iterable, int, int], Iterator],
) -> Callable[[Callable, Iterable, int, int], Iterator]:
    @wraps(_func)
    def inner(
        func: Callable[[S], T],
        it: Iterable[S],
        maxsize: int,
        num_workers: int,
        progress: ProgressT,
    ) -> Iterator:
        it_in = progress.track(it, description="reading")
        it_out = _func(func, it_in, maxsize, num_workers)
        yield from progress.track(it_out, description="processed")

    return inner
