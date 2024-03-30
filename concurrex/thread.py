import threading
from queue import Queue
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from ._thread import map_unordered_semaphore as map_unordered  # noqa: F401
from ._thread_pool import ThreadPool  # noqa: F401
from .utils import Result

T = TypeVar("T")


class ThreadedIterator(Iterator[T]):
    """Use like a normal iterator except that `it` is iterated in a different thread,
    and up to `maxsize` iterations are pre-calculated.
    """

    queue: "Queue[Optional[Result]]"
    exhausted: bool

    def __init__(self, it: Iterable[T], maxsize: int) -> None:
        self.it = it
        self.queue = Queue(maxsize)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.exhausted = False

    def _worker(self) -> None:
        try:
            for item in self.it:
                self.queue.put(Result(result=item))
            self.queue.put(None)
        except Exception as e:
            self.queue.put(Result(exception=e))

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
        except Exception:
            self.thread.join()
            self.exhausted = True
            raise

        return item

    def close(self) -> None:
        self.it.close()

    def send(self, value) -> None:
        self.it.send(value)

    def throw(self, value: BaseException) -> None:
        self.it.throw(value)

    def __iter__(self) -> "ThreadedIterator":
        return self

    def __len__(self) -> int:
        return len(self.it)

    @property
    def buffer(self) -> list:
        with self.queue.mutex:
            return list(self.queue.queue)


class PeriodicExecutor(threading.Thread):
    def __init__(self, func: Callable, delay: float = 1) -> None:
        super().__init__()
        self.func = func
        self.delay = delay
        self._stop = threading.Event()

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()

    def run(self) -> None:
        while not self._stop.wait(self.delay):
            self.func()

    def stop(self):
        self._stop.set()
