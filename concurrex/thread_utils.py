import ctypes
import logging
import threading
from typing import Callable, Optional, Type, Union

from typing_extensions import Self

ExceptHookFuncT = Callable[[threading.ExceptHookArgs], None]


class _Done:
    pass


def kill_thread(thread_id: int, exitcode: int = 1) -> None:
    try:
        from cwinsdk.um.processthreadsapi import TerminateThread

        TerminateThread(thread_id, exitcode)
    except ImportError:
        pass


class ThreadingExceptHook:
    """Context manager to replace the default Python threading exception hook with a custom one.
    When leaving the context the original threading excepthook is restored.

    The custom except hook is passed threading.ExceptHookArgs and the original excepthook.
    """

    def __init__(
        self,
        user_excepthook: Callable[[threading.ExceptHookArgs, ExceptHookFuncT], None],
    ) -> None:
        self.user_excepthook = user_excepthook
        self.old_excepthook: Optional[ExceptHookFuncT] = None

    def new_excepthook(self, args: threading.ExceptHookArgs) -> None:
        self.user_excepthook(args, self.old_excepthook)

    def __enter__(self) -> Self:
        self.old_excepthook = threading.excepthook
        threading.excepthook = self.new_excepthook
        return self

    def __exit__(self, *args):
        threading.excepthook = self.old_excepthook


class MyThread(threading.Thread):
    def raise_exc(self, exception: Type[BaseException]) -> None:
        # https://docs.python.org/3/c-api/init.html#c.PyThreadState_SetAsyncExc
        assert self.native_id is not None
        thread_id = ctypes.c_ulong(self.native_id)

        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(exception))
        if ret == 0:
            raise ValueError("Invalid thread ID")
        elif ret > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def terminate(self, exitcode: int) -> bool:
        if self.is_alive():
            assert self.native_id is not None
            kill_thread(self.native_id, exitcode)
            logging.info("killed thread", self.native_id)
            return True
        return False


class MyBoundedSemaphore(threading.BoundedSemaphore):
    """Extension of `threading.BoundedSemaphore` to support notifying the underlying condition variable
    to wake up threads waiting for this semaphore.
    """

    def notify(self, n: int = 1, blocking: bool = True, timeout: int = -1) -> bool:
        """Return True if threads were notified, and False if a timeout occurred."""

        if not self._cond.acquire(blocking, timeout):
            return False
        try:
            self._cond.notify(n)
        finally:
            self._cond.release()
        return True

    def notify_all(self, blocking: bool = True, timeout: int = -1) -> bool:
        """Return True if threads were notified, and False if a timeout occurred."""

        if not self._cond.acquire(blocking, timeout):
            return False
        try:
            self._cond.notify_all()
        finally:
            self._cond.release()
        return True


class DummySemaphore:
    """A unbounded Semaphore which otherwise offers the same interface as `MyBoundedSemaphore`."""

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

    def notify(self, n: int = 1, blocking: bool = True, timeout: int = -1) -> bool:
        return True

    def notify_all(self, blocking: bool = True, timeout: int = -1) -> bool:
        return True


SemaphoreT = Union[MyBoundedSemaphore, DummySemaphore]


def make_semaphore(n: int) -> SemaphoreT:
    """Return a bounded semaphore for positive n and a unbounded one otherwise."""

    if n > 0:
        return MyBoundedSemaphore(n)
    else:
        return DummySemaphore()


def threading_excepthook(args: threading.ExceptHookArgs, old_excepthook: Callable) -> None:
    exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
    logging.debug("Thread %s interrupted", args.thread, exc_info=exc_info)
    if not isinstance(args.exc_value, KeyboardInterrupt):
        old_excepthook(args)
