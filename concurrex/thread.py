import ctypes
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import FINISHED, Future, _AsCompletedWaiter
from concurrent.futures.thread import BrokenThreadPool, _shutdown, _WorkItem
from functools import wraps
from multiprocessing.pool import ThreadPool as MultiprocessingThreadPool
from queue import Empty, Queue, SimpleQueue
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from genutility.callbacks import Progress as ProgressT
from genutility.concurrency import executor_map, parallel_map
from typing_extensions import TypeAlias

from .utils import SemaphoreT, ThreadingExceptHook, make_semaphore

try:
    from atomicarray import ArrayInt32 as NumArray
except ImportError:
    from .utils import NumArrayPython as NumArray

S = TypeVar("S")
T = TypeVar("T")


class NumTasks(NamedTuple):
    input: int
    processing: int
    output: int


class NoOutstandingResults(Exception):
    """This exception is raised when there are no further tasks in the thread pool."""

    pass


class _Done:
    pass


class _Stop:
    pass


class _Unset:
    pass


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


def threading_excepthook(args: threading.ExceptHookArgs, old_excepthook: Callable) -> None:
    exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
    logging.debug("Thread %s interrupted", args.thread, exc_info=exc_info)
    if not isinstance(args.exc_value, KeyboardInterrupt):
        old_excepthook(args)


def kill_thread(thread_id: int, exitcode: int = 1) -> None:
    try:
        from cwinsdk.um.processthreadsapi import TerminateThread

        TerminateThread(thread_id, exitcode)
    except ImportError:
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
            print("killed thread", self.native_id)
            return True
        return False


class ThreadPoolExecutorWithFuture(ThreadPoolExecutor):
    def submit(self, f, fn, /, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _shutdown:
                raise RuntimeError("cannot schedule new futures after " "interpreter shutdown")

            w = _WorkItem(f, fn, args, kwargs)

            self._work_queue.put(w)
            self._adjust_thread_count()


def _iter_to_queue(it: Iterable[S], q: "Queue[Union[Type[_Done], S]]") -> None:
    for item in it:
        q.put(item)
    q.put(_Done)


def _process_queue(
    func: Callable[[S], T],
    q: "Queue[Union[Type[_Done], S]]",
    futures: "Set[Future[T]]",
    waiter: _AsCompletedWaiter,
    num_workers: int,
) -> None:
    with ThreadPoolExecutorWithFuture(num_workers) as executor:
        while True:
            item = q.get()
            if item is _Done:
                break

            future: "Future[T]" = Future()
            future._waiters.append(waiter)
            futures.add(future)
            executor.submit(future, func, item)

    with waiter.lock:
        waiter.event.set()


def _read_waiter(futures: "Set[Future[T]]", waiter: _AsCompletedWaiter) -> Iterator[Result[T]]:
    while True:
        assert waiter.event.wait(5)
        # print("wait done") # this print uncovers a deadlock
        with waiter.lock:
            finished = waiter.finished_futures
            if not finished:
                break
            waiter.finished_futures = []
            waiter.event.clear()

        for f in finished:
            futures.remove(f)
            with f._condition:
                f._waiters.remove(waiter)

        for f in finished:
            yield Result.from_finished_future(f)


@with_progress
def _map_unordered_exe(
    func: Callable[[S], T],
    it: Iterable[S],
    maxsize: int,
    num_workers: int,
) -> Iterator[Result[T]]:
    """has some race conditions and/or deadlocks"""

    q: "Queue[Union[Type[_Done], S]]" = Queue(maxsize)
    futures: "Set[Future[T]]" = set()
    waiter = _AsCompletedWaiter()

    threading.Thread(target=_iter_to_queue, args=(it, q)).start()
    threading.Thread(target=_process_queue, args=(func, q, futures, waiter, num_workers)).start()

    yield from _read_waiter(futures, waiter)


def _map_queue(
    func: Callable[[S], T],
    in_q: "Queue[Union[Type[_Done], S]]",
    out_q: "Queue[Optional[Result[T]]]",
) -> None:
    while True:
        item = in_q.get()
        if item is _Done:
            break
        out_q.put(Result.from_func(func, item))
    out_q.put(None)


def _read_queue_update_total(
    it: Iterable[S],
    in_q: "Queue[Union[Type[_Done], S]]",
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
    description: str = "reading",
) -> None:
    for item in progress.track(it, description=description):
        update["total"] += 1
        in_q.put(item)
    for _ in range(num_workers):
        in_q.put(_Done)


def _read_out_queue(
    out_q: "Queue[Optional[Result[T]]]",
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
    description: str = "processed",
) -> Iterator[Result[T]]:
    with progress.task(total=update["total"], description=description) as task:
        completed = 0
        task.update(completed=completed, **update)
        while True:
            item = out_q.get()
            if item is None:
                num_workers -= 1
                if num_workers == 0:
                    break
            else:
                completed += 1
                task.update(completed=completed, **update)
                yield item


def _map_unordered_bq(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    in_q: "Queue[Union[Type[_Done], S]]" = Queue(maxsize)
    out_q: "Queue[Optional[Result[T]]]" = Queue(maxsize)
    update = {"total": 0}

    threading.Thread(target=_read_queue_update_total, args=(it, in_q, update, num_workers, progress)).start()
    for _ in range(num_workers):
        threading.Thread(target=_map_queue, args=(func, in_q, out_q)).start()

    yield from _read_out_queue(out_q, update, num_workers, progress)


def _read_queue_update_total_semaphore(
    it: Iterable[S],
    in_q: "SimpleQueue[Union[Type[_Done], S]]",
    semaphore: SemaphoreT,
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
    description: str = "reading",
) -> None:
    try:
        for item in progress.track(it, description=description):
            semaphore.acquire()  # notifying it allows waiting exceptions to interrupt it
            update["total"] += 1
            in_q.put(item)
    except KeyboardInterrupt:
        while True:
            try:
                in_q.get_nowait()
            except Empty:
                break
    finally:
        for _ in range(num_workers):
            in_q.put(_Done)


def _read_out_queue_semaphore(
    out_q: "SimpleQueue[Optional[Result[T]]]",
    update: Dict[str, int],
    semaphore: SemaphoreT,
    num_workers: int,
    progress: ProgressT,
    description: str = "processed",
) -> Iterator[Result[T]]:
    with progress.task(total=update["total"], description=description) as task:
        completed = 0
        task.update(completed=completed, **update)
        while True:
            item = out_q.get()
            if item is None:
                num_workers -= 1
                if num_workers == 0:
                    break
            else:
                completed += 1
                task.update(completed=completed, **update)
                yield item
                semaphore.release()


def _map_unordered_sem(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    assert maxsize >= num_workers

    in_q: "SimpleQueue[Union[Type[_Done], S]]" = SimpleQueue()
    out_q: "SimpleQueue[Optional[Result[T]]]" = SimpleQueue()
    update = {"total": 0}
    semaphore = make_semaphore(maxsize)
    threads: List[MyThread] = []

    t_read = MyThread(
        target=_read_queue_update_total_semaphore,
        name="task-reader",
        args=(it, in_q, semaphore, update, num_workers, progress),
    )
    t_read.start()
    threads.append(t_read)
    for _ in range(num_workers):
        t = MyThread(target=_map_queue, args=(func, in_q, out_q))
        t.start()
        threads.append(t)

    with ThreadingExceptHook(threading_excepthook):
        try:
            yield from _read_out_queue_semaphore(out_q, update, semaphore, num_workers, progress)
        except (KeyboardInterrupt, GeneratorExit):
            # logging.debug("Caught %s, trying to clean up", type(e).__name__)
            t_read.raise_exc(KeyboardInterrupt)
            if not semaphore.notify_all(timeout=10):  # this can deadlock
                raise RuntimeError("deadlock")

            for thread in threads:
                thread.join()
            any_terminated = False
            for thread in threads:
                any_terminated = any_terminated or thread.terminate(1)
            if any_terminated:
                raise RuntimeError("Terminated blocking threads")
            raise
        except BaseException as e:
            logging.error("Caught %s, trying to clean up", type(e).__name__)
            raise


def _queue_reader(q: "Queue[Optional[Future[T]]]") -> Iterator[Result[T]]:
    """requires active executor"""

    while True:
        item = q.get()
        if item is None:
            break
        yield Result.from_future(item)


def _submit_from_queue(
    func: Callable[[S], T],
    it: Iterable[S],
    ex: ThreadPoolExecutor,
    q: "Queue[Optional[Future[T]]]",
) -> None:
    for item in it:
        future = ex.submit(func, item)
        q.put(future)
    q.put(None)


@with_progress
def executor_ordered(
    func: Callable[[S], T],
    it: Iterable[S],
    maxsize: int,
    num_workers: int,
) -> Iterator[Result[T]]:
    q: "Queue[Optional[Future[T]]]" = Queue(maxsize)
    with ThreadPoolExecutor(num_workers) as ex:
        threading.Thread(target=_submit_from_queue, args=(func, it, ex, q)).start()
        yield from _queue_reader(q)


def result_wrapper(func: Callable):
    @wraps(func)
    def inner(*args, **kwargs):
        return Result.from_func(func, *args, **kwargs)

    return inner


@with_progress
def parallel_map_thread_unordered(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator:
    yield from parallel_map(
        result_wrapper(func),
        it,
        poolcls=MultiprocessingThreadPool,
        ordered=False,
        parallel=True,
        bufsize=maxsize,
        workers=num_workers,
    )


@with_progress
def parallel_map_thread_ordered(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator:
    yield from parallel_map(
        result_wrapper(func),
        it,
        poolcls=MultiprocessingThreadPool,
        ordered=True,
        parallel=True,
        bufsize=maxsize,
        workers=num_workers,
    )


@with_progress
def executor_map_thread_unordered(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator:
    for f in executor_map(
        func,
        it,
        executercls=ThreadPoolExecutor,
        ordered=False,
        parallel=True,
        bufsize=maxsize,
        workers=num_workers,
    ):
        yield Result.from_finished_future(f)


@with_progress
def executor_map_thread_ordered(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator:
    for f in executor_map(
        func,
        it,
        executercls=ThreadPoolExecutor,
        ordered=True,
        parallel=True,
        bufsize=maxsize,
        workers=num_workers,
    ):
        yield Result.from_future(f)


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

    @property
    def buffer(self) -> list:
        with self.queue.mutex:
            return list(self.queue.queue)


WorkQueueItemT: TypeAlias = "Union[Type[_Done], Type[_Stop], Tuple[Callable[[S], T], tuple, dict]]"
WorkQueueT: TypeAlias = "Union[SimpleQueue[WorkQueueItemT], Queue[WorkQueueItemT]]"
ResultQueueItemT: TypeAlias = "Optional[Result[T]]"
ResultQueueT: TypeAlias = "Union[SimpleQueue[ResultQueueItemT], Queue[ResultQueueItemT]]"


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
                    self._counts += counts_before
                    func, args, kwargs = item
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
        print("ThreadPool.close")
        for _ in range(self.num_workers):
            # stop worker threads
            self.in_q.put(_Stop)
        if self.signal_threads():
            print("some threads still active")
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


def _map_unordered_tp(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    with ThreadPool(num_workers, progress) as tp:
        yield from tp.map_unordered(func, it, maxsize)
