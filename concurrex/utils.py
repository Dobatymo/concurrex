import threading
from concurrent.futures._base import FINISHED, Future
from functools import wraps
from typing import Callable, Generic, Iterable, Iterator, Optional, Tuple, Type, TypeVar, Union

from genutility.callbacks import Progress as ProgressT

T = TypeVar("T")
S = TypeVar("S")


class _Unset:
    pass


class Result(Generic[T]):
    __slots__ = ("result", "exception")

    def __init__(
        self,
        result: Union[Type[_Unset], T] = _Unset,
        exception: Optional[Exception] = None,
    ) -> None:
        self.result = result
        self.exception = exception

    def __eq__(self, other) -> bool:
        return (self.result, self.exception) == (other.result, other.exception)

    def __lt__(self, other) -> bool:
        return (self.result, self.exception) < (other.result, other.exception)

    def get(self) -> T:
        if self.exception is not None:
            raise self.exception
        assert self.result is not _Unset
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
    def from_func(cls, func: Callable, *args, **kwargs) -> "Result[T]":
        try:
            return cls(result=func(*args, **kwargs))
        except Exception as e:
            return cls(exception=e)


class CvWindow:
    def __init__(self, name: Optional[str] = None) -> None:
        import cv2

        self.name = name or str(id(self))
        self.cv2 = cv2

    def show(self, image, title: Optional[str] = None) -> None:
        self.cv2.imshow(self.name, image)
        if title is not None:
            self.cv2.setWindowTitle(self.name, title)
        self.cv2.waitKey(1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cv2.destroyWindow(self.name)


class NumArrayPython(Generic[T]):
    def __init__(self, *args: T):
        self._arr = list(args)
        self._lock = threading.Lock()

    def __iadd__(self, other: "NumArrayPython") -> "NumArrayPython":
        with self._lock:
            for i in range(len(self._arr)):
                self._arr[i] += other._arr[i]
        return self

    def __isub__(self, other: "NumArrayPython") -> "NumArrayPython":
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

    def __iadd__(self, other: NumArrayAtomicsInt) -> "NumArrayAtomics":
        self.a.fetch_add(other.val)
        return self

    def __isub__(self, other: NumArrayAtomicsInt) -> "NumArrayAtomics":
        self.a.fetch_sub(other.val)
        return self

    def to_tuple(self) -> Tuple[int, ...]:
        return tuple(self)

    def __iter__(self) -> Iterator[int]:
        rem, c = divmod(self.a.load(), 2**16)
        a, b = divmod(rem, 2**16)
        return iter([a, b, c])


def with_progress(_func):
    @wraps(_func)
    def inner(
        func: Callable[[S], T],
        it: Iterable[S],
        maxsize: int,
        num_workers: int,
        progress: ProgressT,
    ):
        it_in = progress.track(it, description="reading")
        it_out = _func(func, it_in, maxsize, num_workers)
        yield from progress.track(it_out, description="processed")

    return inner
