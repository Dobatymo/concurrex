import logging
import os
import threading
from queue import Empty, Queue, SimpleQueue
from typing import Callable, Generic, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union

from genutility.callbacks import Progress as ProgressT
from typing_extensions import TypeAlias

from .thread_utils import MyThread, SemaphoreT, ThreadingExceptHook, _Done, make_semaphore, threading_excepthook
from .utils import Result

try:
    from atomicarray import ArrayInt32 as NumArray
except ImportError:
    from .utils import NumArrayPython as NumArray

S = TypeVar("S")
T = TypeVar("T")

WorkQueueItemT: TypeAlias = "Union[Type[_Done], Type[_Stop], Tuple[Callable[[S], T], tuple, dict]]"
WorkQueueT: TypeAlias = "Union[SimpleQueue[WorkQueueItemT], Queue[WorkQueueItemT]]"
ResultQueueItemT: TypeAlias = "Optional[Result[T]]"
ResultQueueT: TypeAlias = "Union[SimpleQueue[ResultQueueItemT], Queue[ResultQueueItemT]]"


class NumTasks(NamedTuple):
    input: int
    processing: int
    output: int


class NoOutstandingResults(Exception):
    """This exception is raised when there are no further tasks in the thread pool."""

    pass


class _Stop:
    pass


class Executor(Generic[T]):
    def __init__(self, threadpool: "ThreadPool", bufsize: int = 0) -> None:
        self.threadpool = threadpool
        self.semaphore = make_semaphore(bufsize)
        assert not self.threadpool.signal_threads()

    def execute(self, func: Callable[[S], T], *args, **kwargs) -> None:
        """Runs `func` in a worker thread and returns"""
        threadpool = self.threadpool

        self.semaphore.acquire()
        threadpool._counts += NumArray(1, 0, 0)
        threadpool.total += 1
        threadpool.in_q.put((func, args, kwargs))

    def done(self) -> None:
        for _ in range(self.threadpool.num_workers):
            self.threadpool.in_q.put(_Done)

    def iter_unordered(self, wait_done: bool = False, description: str = "reading") -> Iterator[Result[T]]:
        threadpool = self.threadpool
        semaphore = self.semaphore

        out_q = threadpool.out_q
        num_workers = threadpool.num_workers  # copy

        with threadpool.progress.task(total=threadpool.total, description=description) as task:
            completed = 0
            task.update(completed=completed, total=threadpool.total)
            counts = NumArray(0, 0, -1)
            while True:
                if wait_done:
                    item = out_q.get()
                    if item is None:
                        num_workers -= 1
                        if num_workers == 0:
                            break
                        continue
                    else:
                        semaphore.release()
                else:
                    try:
                        semaphore.release()
                    except ValueError:
                        break
                    item = out_q.get()
                    assert item is not None

                completed += 1
                yield item

                threadpool._counts += counts
                task.update(completed=completed, total=threadpool.total)

    def get_unordered(self, wait_done: bool = False) -> T:
        num_workers = self.threadpool.num_workers  # copy
        out_q = self.threadpool.out_q

        if wait_done:
            while True:
                item = out_q.get()
                if item is None:
                    num_workers -= 1
                    if num_workers == 0:
                        raise NoOutstandingResults()
                else:
                    self.semaphore.release()
                    break
        else:
            try:
                self.semaphore.release()
            except ValueError:
                raise NoOutstandingResults()
            item = out_q.get()
            assert item is not None  # for mypy

        self.threadpool._counts += NumArray(0, 0, -1)
        return item.get()


class ThreadPool(Generic[T]):
    num_workers: int

    def __init__(self, num_workers: Optional[int] = None, progress: Optional[ProgressT] = None) -> None:
        self.num_workers = num_workers or os.cpu_count() or 1
        self.progress = progress or ProgressT()
        self._counts = NumArray(0, 0, 0)

        self.total = 0
        self.in_q: WorkQueueT = SimpleQueue()
        self.out_q: ResultQueueT = SimpleQueue()
        self.events = [threading.Event() for _ in range(self.num_workers)]
        self.threads = [MyThread(target=self._map_queue, args=(self.in_q, self.out_q, event)) for event in self.events]

        for t in self.threads:
            t.start()

    def signal_threads(self) -> List[MyThread]:
        """Returns threads which where already signaled before"""

        out: List[MyThread] = []
        for event, thread in zip(self.events, self.threads):
            if event.is_set():
                out.append(thread)
            event.set()
        return out

    def _map_queue(
        self,
        in_q: WorkQueueT,
        out_q: ResultQueueT,
        event: threading.Event,
    ) -> None:
        counts_before = NumArray(-1, 1, 0)
        counts_after = NumArray(0, -1, 1)
        while event.wait():
            while True:
                item = in_q.get()

                if item is _Done:
                    out_q.put(None)
                    event.clear()
                    break
                elif item is _Stop:
                    return
                else:
                    func, args, kwargs = item
                    self._counts += counts_before
                    out_q.put(Result.from_func(func, *args, **kwargs))
                    self._counts += counts_after

    def _read_it(
        self,
        it: Iterable[Tuple[Callable[[S], T], tuple, dict]],
        total: Optional[int],
        semaphore: SemaphoreT,
        description: str = "reading",
    ) -> None:
        try:
            # read items from iterable to queue
            counts = NumArray(1, 0, 0)
            for item in self.progress.track(it, total=total, description=description):
                semaphore.acquire()  # notifying it allows waiting exceptions to interrupt it
                self._counts += counts
                self.total += 1
                self.in_q.put(item)
        except KeyboardInterrupt:
            self.drain_input_queue()
        finally:
            # add _Done values to input queue, so workers can recognize when the iterable is exhausted
            for _ in range(self.num_workers):
                self.in_q.put(_Done)

    def _read_queue(
        self,
        semaphore: SemaphoreT,
        description: str = "processed",
    ) -> Iterator[Result[T]]:
        num_workers = self.num_workers
        with self.progress.task(total=self.total, description=description) as task:
            completed = 0
            task.update(completed=completed, total=self.total)
            counts = NumArray(0, 0, -1)
            while True:
                item = self.out_q.get()
                if item is None:
                    num_workers -= 1
                    if num_workers == 0:
                        break
                else:
                    completed += 1
                    task.update(completed=completed, total=self.total)
                    yield item
                    semaphore.release()
                    self._counts += counts

    def drain_input_queue(self) -> None:
        counts = NumArray(-1, 0, 0)
        while True:
            try:
                item = self.in_q.get_nowait()
                if item is not _Done and item is not _Stop:
                    self._counts += counts
            except Empty:
                break

    def num_tasks(self) -> NumTasks:
        return NumTasks(*self._counts.to_tuple())

    def executor(self, bufsize: int = 0) -> Executor[T]:
        """bufsize should be set to 0 when tasks are submitted and retrieved by the same thread,
        otherwise it will deadlock when more bufsize tasks are queued.
        When results are retrieved by a different thread,
        it should be set to >0 to avoid growing the queue without limit.
        """

        return Executor(self, bufsize)

    def _run_iter(
        self,
        it: Iterable[Tuple[Callable[[S], T], tuple, dict]],
        total: Optional[int],
        bufsize: int = 0,
    ) -> Iterator[Result[T]]:
        semaphore = make_semaphore(bufsize)
        t_read = MyThread(
            target=self._read_it,
            name="task-reader",
            args=(it, total, semaphore),
        )
        t_read.start()

        assert not self.signal_threads()

        with ThreadingExceptHook(threading_excepthook):
            try:
                yield from self._read_queue(semaphore)
            except (KeyboardInterrupt, GeneratorExit) as e:
                logging.warning("Caught %s, trying to clean up", type(e).__name__)
                t_read.raise_exc(KeyboardInterrupt)
                if not semaphore.notify_all(timeout=10):  # this can deadlock
                    raise RuntimeError("either deadlocked or worker tasks didn't complete fast enough")
                raise
            except BaseException as e:
                logging.error("Caught %s, trying to clean up", type(e).__name__)
                raise

        t_read.join()

    def __enter__(self) -> "ThreadPool[T]":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self.drain_input_queue()
        self.close()

    def close(self) -> None:
        logging.debug("ThreadPool.close")
        for _ in range(self.num_workers):
            # stop worker threads
            self.in_q.put(_Stop)
        if self.signal_threads():
            logging.warning("some threads still active")
        for thread in self.threads:
            # wait for worker threads to stop
            thread.join()

    def map_unordered(self, func: Callable[[S], T], it: Iterable[S], bufsize: int = 0) -> Iterator[Result[T]]:
        _it = ((func, (i,), {}) for i in it)
        try:
            total = len(it)
        except TypeError:
            total = None
        return self._run_iter(_it, total, bufsize)

    def starmap_unordered(self, func: Callable[[S], T], it: Iterable[tuple], bufsize: int = 0) -> Iterator[Result[T]]:
        _it = ((func, args, {}) for args in it)
        try:
            total = len(it)
        except AttributeError:
            total = None
        return self._run_iter(_it, total, bufsize)


def map_unordered_concurrex(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    with ThreadPool(num_workers, progress) as tp:
        yield from tp.map_unordered(func, it, maxsize)
