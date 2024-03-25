import threading
from typing import Callable, Generic, Iterator, Optional, Tuple, TypeVar, Union

T = TypeVar("T")
ExceptHookFuncT = Callable[[threading.ExceptHookArgs], None]


class ThreadingExceptHook:
    def __init__(
        self,
        user_excepthook: Callable[[threading.ExceptHookArgs, ExceptHookFuncT], None],
    ) -> None:
        self.user_excepthook = user_excepthook
        self.old_excepthook: Optional[ExceptHookFuncT] = None

    def new_excepthook(self, args: threading.ExceptHookArgs) -> None:
        self.user_excepthook(args, self.old_excepthook)

    def __enter__(self) -> "ThreadingExceptHook":
        self.old_excepthook = threading.excepthook
        threading.excepthook = self.new_excepthook
        return self

    def __exit__(self, *args):
        threading.excepthook = self.old_excepthook


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


class MyBoundedSemaphore(threading.BoundedSemaphore):
    def notify(self, n: int = 1) -> None:
        with self._cond:
            self._cond.notify(n)

    def notify_all(self, blocking: bool = True, timeout: int = -1) -> bool:
        if not self._cond.acquire(blocking, timeout):
            return False
        try:
            self._cond.notify_all()
        finally:
            self._cond.release()
        return True


class DummySemaphore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def acquire(self, blocking=True, timeout=None) -> None:
        with self._lock:
            self._value += 1

    def release(self, n=1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        with self._lock:
            if self._value - n < 0:
                raise ValueError("Semaphore released too many times")
            self._value -= n

    def notify(self, n: int = 1) -> None:
        pass

    def notify_all(self, blocking: bool = True, timeout: int = -1) -> bool:
        return True


SemaphoreT = Union[MyBoundedSemaphore, DummySemaphore]


def make_semaphore(n: int) -> SemaphoreT:
    if n > 0:
        return MyBoundedSemaphore(n)
    else:
        return DummySemaphore()


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

    def __iter__(self) -> Iterator[int]:
        rem, c = divmod(self.a.load(), 2**16)
        a, b = divmod(rem, 2**16)
        return iter([a, b, c])
