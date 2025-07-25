import logging
import os
import threading
from queue import Empty, Queue, SimpleQueue
from typing import Callable, Generic, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union

from genutility.callbacks import Progress as ProgressT
from typing_extensions import Self, TypeAlias

from .thread_utils import MyThread, SemaphoreT, ThreadingExceptHook, _Done, make_semaphore, threading_excepthook
from .utils import TOTAL_TIMEOUT, Result, debug_join, debug_wait, get_extra

try:
    from atomicarray import ArrayInt32 as NumArray
except ImportError:
    from .utils import NumArrayPython as NumArray

logger = logging.getLogger(__name__)

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
        active_threads = self.threadpool._signal_threads()
        if active_threads:
            raise RuntimeError(f"{active_threads} threads already active")

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

    def iter_unordered(
        self, wait_done: bool = False, description: str = "reading", transient: bool = False
    ) -> Iterator[Result[T]]:
        threadpool = self.threadpool
        semaphore = self.semaphore

        out_q = threadpool.out_q
        num_workers = threadpool.num_workers  # copy

        with threadpool.progress.task(total=threadpool.total, description=description, transient=transient) as task:
            completed = 0
            task.update(completed=completed, total=threadpool.total)
            counts = NumArray(0, 0, -1)
            while True:
                if wait_done:
                    item = out_q.get()
                    if item is None:
                        num_workers -= 1
                        logger.debug("Queue empty. %d workers remaining.", num_workers, extra=get_extra(self))
                        if num_workers == 0:
                            break
                        continue
                    else:
                        semaphore.release()
                else:
                    try:
                        semaphore.release()
                    except ValueError as e:
                        logger.debug("Releasing %s failed: %s", semaphore, e, extra=get_extra(self))
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
                    logger.debug("Queue empty. %d workers remaining.", num_workers, extra=get_extra(self))
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

    # fixme: executor could also return a context manager which calls done() on exit


class ThreadPool(Generic[T]):
    num_workers: int

    def __init__(
        self, num_workers: Optional[int] = None, progress: Optional[ProgressT] = None, daemon: Optional[bool] = None
    ) -> None:
        self.num_workers = num_workers or os.cpu_count() or 1
        self.progress = progress or ProgressT()
        self.daemon = daemon

        self._counts = NumArray(0, 0, 0)

        self.total = 0
        self.in_q: WorkQueueT = SimpleQueue()
        self.out_q: ResultQueueT = SimpleQueue()
        self.events_a = [threading.Event() for _ in range(self.num_workers)]
        self.events_b = [threading.Event() for _ in range(self.num_workers)]
        self.threads = [
            MyThread(
                target=self._map_queue,
                name=f"ThreadPool-{id(self):x}-map_queue-{i}",
                args=(self.in_q, self.out_q, event_a, event_b),
                daemon=self.daemon,
            )
            for i, (event_a, event_b) in enumerate(zip(self.events_a, self.events_b))
        ]

        for t in self.threads:
            t.start()

    def _wait_threads_idle_or_dead(self) -> None:
        """Wait until threadpool is either idle or all threads are terminated."""

        debug_wait(self.events_b, TOTAL_TIMEOUT, extra=get_extra(self))

    def _signal_threads(self) -> List[MyThread]:
        """Returns threads which where already signaled before"""

        out: List[MyThread] = []
        for event_a, thread in zip(self.events_a, self.threads):
            if event_a.is_set():
                out.append(thread)
            event_a.set()
        return out

    def _map_queue(
        self,
        in_q: WorkQueueT,
        out_q: ResultQueueT,
        event_a: threading.Event,  # ready to consume tasks when set
        event_b: threading.Event,  # idle when set
    ) -> None:
        try:
            event_b.set()
            counts_before = NumArray(-1, 1, 0)
            counts_after = NumArray(0, -1, 1)
            while True:
                debug_wait([event_a], TOTAL_TIMEOUT, extra=get_extra(self))
                event_b.clear()
                while True:
                    item = in_q.get()
                    if item is _Done:
                        out_q.put(None)
                        event_a.clear()
                        event_b.set()
                        break
                    elif item is _Stop:
                        event_a.clear()
                        event_b.set()
                        return
                    else:
                        func, args, kwargs = item
                        self._counts += counts_before
                        out_q.put(Result.from_func(func, *args, **kwargs))
                        self._counts += counts_after
        finally:
            logger.debug("Thread exiting", extra=get_extra(self))

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
            num = self._drain_input_queue()
            logger.debug("Interrupted, drained input queue (%d items)", num, extra=get_extra(self))
        finally:
            # add _Done values to input queue, so workers can recognize when the iterable is exhausted
            for _ in range(self.num_workers):
                self.in_q.put(_Done)

            logger.debug("Thread exiting", extra=get_extra(self))

    def _read_queue(
        self,
        semaphore: SemaphoreT,
        description: str = "processed",
    ) -> Iterator[Result[T]]:
        """running in main thread"""

        num_workers = self.num_workers
        with self.progress.task(total=self.total, description=description) as task:
            completed = 0
            task.update(completed=completed, total=self.total)
            counts = NumArray(0, 0, -1)
            while True:
                item = self.out_q.get()
                if item is None:
                    num_workers -= 1
                    logger.debug("Queue empty. %d workers remaining.", num_workers, extra=get_extra(self))
                    if num_workers == 0:
                        break
                else:
                    completed += 1
                    task.update(completed=completed, total=self.total)
                    yield item
                    semaphore.release()
                    self._counts += counts

    def _drain_input_queue(self) -> int:
        counts = NumArray(-1, 0, 0)
        out = 0
        while True:
            try:
                item = self.in_q.get_nowait()
                out += 1
                if item is not _Done and item is not _Stop:
                    self._counts += counts
            except Empty:
                break
        return out

    def num_tasks(self) -> NumTasks:
        return NumTasks(*self._counts.to_tuple())

    def executor(self, bufsize: int = 0) -> Executor[T]:
        """bufsize should be set to 0 when tasks are submitted and retrieved by the same thread,
        otherwise it will deadlock when more than bufsize tasks are queued.
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
        """running in main thread"""

        semaphore = make_semaphore(bufsize)
        t_read = MyThread(
            target=self._read_it,
            name=f"ThreadPool-{id(self):x}-read_it",
            args=(it, total, semaphore),
            daemon=self.daemon,
        )
        t_read.start()

        # start all waiting reader threads and raise if some were already running
        active_threads = self._signal_threads()
        if active_threads:
            raise RuntimeError(f"{active_threads} threads already active")

        with ThreadingExceptHook(threading_excepthook):
            try:
                yield from self._read_queue(semaphore)
            except (KeyboardInterrupt, GeneratorExit) as e:
                logger.debug("Caught %s, trying to clean up", type(e).__name__, extra=get_extra(self))
                t_read.raise_exc(KeyboardInterrupt)
                if not semaphore.notify_all(timeout=10):  # this can deadlock
                    raise RuntimeError("either deadlocked or worker tasks didn't complete fast enough")
                raise
            except BaseException as e:
                logger.error("Caught %s, trying to clean up", type(e).__name__, extra=get_extra(self))
                # fixme: we should probably try to stop t_read here as well
                raise
            finally:
                debug_join([t_read], TOTAL_TIMEOUT, extra=get_extra(self))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            num = self._drain_input_queue()
            logger.debug("Caught %s, drained input queue (%d items)", exc_type.__name__, num, extra=get_extra(self))
        self.close()

    def close(self) -> None:
        logger.debug(
            "Closing. in-queue-size=%d, out-queue-size=%d", self.in_q.qsize(), self.out_q.qsize(), extra=get_extra(self)
        )
        for _ in range(self.num_workers):
            # message worker threads to stop
            self.in_q.put(_Stop)

        self._wait_threads_idle_or_dead()
        # if all threads are idle here, signaling them will cause them to handle the _Stop marker and terminate
        # if all threads are already terminated, signaling them will do nothing
        # signaling an already active thread might lead to a deadlock, so waiting first is important
        active_threads = self._signal_threads()
        if active_threads:
            logger.warning("%d threads already active", len(active_threads), extra=get_extra(self))
            # raise RuntimeError(f"{active_threads} threads already active")

        debug_join(self.threads, TOTAL_TIMEOUT, extra=get_extra(self))

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
    daemon: Optional[bool] = True,
) -> Iterator[Result[T]]:
    tp_cm: ThreadPool[T] = ThreadPool(num_workers, progress, daemon)
    with tp_cm as tp:
        yield from tp.map_unordered(func, it, maxsize)
