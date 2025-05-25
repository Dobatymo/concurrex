import logging
import threading
from queue import Queue
from typing import Any, Callable, Iterable, Iterator, List, Optional, TypeVar

from typing_extensions import ParamSpec, Self

from ._thread import map_unordered_semaphore as map_unordered  # noqa: F401
from ._thread_pool import ThreadPool  # noqa: F401
from .utils import Result, get_extra

T = TypeVar("T")
P = ParamSpec("P")


class ThreadedIterator(Iterator[T]):
    """Use like a normal iterator except that `it` is iterated in a different thread,
    and up to `maxsize` iterations are pre-calculated.
    """

    queue: "Queue[Optional[Result]]"
    exhausted: bool

    def __init__(self, it: Iterable[T], maxsize: int) -> None:
        """Run `it` in another thread.
        If `maxsize` is less than or equal to zero, the queue size is infinite.
        """

        self.it = it
        self._count = 0
        self._lock = threading.Lock()
        self.queue = Queue(maxsize)
        self.thread = threading.Thread(target=self._worker, name=f"ThreadedIterator-{id(self):x}", daemon=True)
        self.thread.start()
        self.exhausted = False

    def _worker(self) -> None:
        it = iter(self.it)
        try:
            while True:
                with self._lock:
                    item = next(it)
                self.queue.put(Result(result=item))
                self._count += 1
        except StopIteration:
            self.queue.put(None)
        except Exception as e:
            self.queue.put(Result(exception=e))
        except BaseException as e:
            self.queue.put(Result(exception=e))
            raise
        finally:
            logging.debug("Thread for %r exiting", self.it, extra=get_extra(self))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args):
        self.close()

    def __next__(self) -> T:
        if self.exhausted:
            raise StopIteration

        result = self.queue.get()
        if result is None:
            self.thread.join()
            self.exhausted = True
            raise StopIteration

        try:
            item = result.get()
        except BaseException:
            # this might timeout if a KeyboardInterrupt is raised in this thread in the try block above
            # it's fine when it's raised in the worker thread
            self.thread.join(timeout=1)
            if self.thread.is_alive():
                logging.error("Thread join timed out", extra=get_extra(self))
            self.exhausted = True
            raise

        return item

    def close(self) -> List[Result[T]]:
        if self.exhausted:
            return []

        with self._lock:
            self.it.close()

        out = []
        while True:
            result = self.queue.get()  # unblock the thread
            if result is None:
                break
            out.append(result)

        self.thread.join()
        self.exhausted = True
        return out

    def send(self, value) -> None:
        with self._lock:
            self.it.send(value)

    def throw(self, value: BaseException) -> None:
        with self._lock:
            self.it.throw(value)

    def __iter__(self) -> Self:
        return self

    def __len__(self) -> int:
        return len(self.it)

    def processed(self) -> int:
        return self._count

    def _wait_for_queue_full(self) -> None:
        """only ever returns when queue is full, so it can deadlock if the queue never gets full"""

        if self.queue.maxsize < 1:
            raise RuntimeError("Can only wait on bounded queues")

        with self.queue.not_empty:
            while self.queue._qsize() < self.queue.maxsize:
                self.queue.not_empty.wait()
            return

    @property
    def buffer(self) -> List[Result[T]]:
        with self.queue.mutex:
            return [result for result in self.queue.queue if result is not None]  # queue.queue is a deque


class PeriodicExecutor(threading.Thread):
    def __init__(self, delay: float, func: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> None:
        """Runs func(*args, **kwargs) every `delay` seconds.
        The first call is after `delay` seconds.
        """

        super().__init__(name=f"PeriodicExecutor-{id(self):x}", daemon=None)
        self.delay = delay
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._stop = threading.Event()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def run(self) -> None:
        try:
            while not self._stop.wait(self.delay):
                self.func(*self.args, **self.kwargs)
        finally:
            logging.debug("Thread for %r exiting", self.func, extra=get_extra(self))

    def stop(self) -> None:
        self._stop.set()
