import ctypes
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import FINISHED, Future, _AsCompletedWaiter
from concurrent.futures.thread import BrokenThreadPool, _shutdown, _WorkItem
from functools import wraps
from itertools import count
from multiprocessing.pool import ThreadPool
from queue import Empty, Queue, SimpleQueue
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Set, Type, TypeVar, Union

from genutility.callbacks import Progress as ProgressT
from genutility.concurrency import executor_map, parallel_map
from genutility.func import identity
from genutility.rich import Progress
from genutility.time import PrintStatementTime
from rich.progress import Progress as RichProgress

S = TypeVar("S")
T = TypeVar("T")


class _Done:
    pass


class Result(Generic[T]):
    __slots__ = ("result", "exception")

    def __init__(self, result: Optional[T] = None, exception: Optional[Exception] = None) -> None:
        self.result = result
        self.exception = exception

    def get(self) -> T:
        if self.exception is not None:
            raise self.exception
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
            result = f.result()
            exception = None
        except Exception:
            result = None
            exception = f._exception
        return cls(result, exception)

    @classmethod
    def from_func(cls, func: Callable, *args, **kwargs) -> "Result[T]":
        try:
            result = func(*args, **kwargs)
            exception = None
        except Exception as e:
            result = None
            exception = e
        return Result(result, exception)


class MyThread(threading.Thread):
    def raise_exc(self, exception: BaseException) -> None:
        # https://docs.python.org/3/c-api/init.html#c.PyThreadState_SetAsyncExc

        thread_id = ctypes.c_ulong(self.native_id)

        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(exception))
        if ret == 0:
            raise ValueError("Invalid thread ID")
        elif ret > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")


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
    waiter,
    num_workers: int,
) -> None:
    with ThreadPoolExecutorWithFuture(num_workers) as executor:
        while True:
            item = q.get()
            if item is _Done:
                break

            future = Future()
            future._waiters.append(waiter)
            futures.add(future)
            executor.submit(future, func, item)

    with waiter.lock:
        waiter.event.set()


def _read_waiter(futures: "Set[Future[T]]", waiter) -> Iterator[Result[T]]:
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


def process(
    func: Callable[[S], T],
    it: Iterable[S],
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    """has some race conditions and/or deadlocks"""

    q = Queue(maxsize)
    futures = set()
    waiter = _AsCompletedWaiter()

    threading.Thread(target=_iter_to_queue, args=(it, q)).start()
    threading.Thread(target=_process_queue, args=(func, q, futures, waiter, num_workers)).start()

    yield from progress.track(_read_waiter(futures, waiter))


def _map_queue(
    func: Callable[[S], T],
    in_q: "Queue[Union[Type[_Done], S]]",
    out_q: "Queue[Optional[Result[T]]]",
) -> None:
    while True:
        while True:
            try:
                item = in_q.get(timeout=1)
                break
            except Empty:
                pass
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
) -> None:
    for item in progress.track(it, description="reading"):
        update["total"] += 1
        in_q.put(item)
    for _ in range(num_workers):
        in_q.put(_Done)


def _read_out_queue(
    out_q: "Queue[Optional[Result[T]]]",
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    with progress.task(description="processed") as task:
        for i in count(1):
            item = out_q.get()
            if item is None:
                num_workers -= 1
                if num_workers == 0:
                    break
            task.update(completed=i, **update)
            yield item


def process_2(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    in_q = Queue(maxsize)
    out_q = Queue(maxsize)
    update = {"total": 0}

    threading.Thread(target=_read_queue_update_total, args=(it, in_q, update, num_workers, progress)).start()
    for _ in range(num_workers):
        threading.Thread(target=_map_queue, args=(func, in_q, out_q)).start()

    yield from _read_out_queue(out_q, update, num_workers, progress)


def _read_queue_update_total_semaphore(
    it: Iterable[S],
    in_q: "SimpleQueue[Union[Type[_Done], S]]",
    semaphore: threading.Semaphore,
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
) -> None:
    for item in progress.track(it, description="reading"):
        while True:
            if semaphore.acquire(timeout=1):
                break
        update["total"] += 1
        in_q.put(item)
    for _ in range(num_workers):
        in_q.put(_Done)


def _read_out_queue_semaphore(
    out_q: "SimpleQueue[Optional[Result[T]]]",
    update: Dict[str, int],
    semaphore: threading.Semaphore,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    with progress.task(description="processed") as task:
        for i in count(1):
            item = out_q.get()
            if item is None:
                num_workers -= 1
                if num_workers == 0:
                    break
            else:
                task.update(completed=i, **update)
                yield item
                semaphore.release()


def process_3(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    assert maxsize >= num_workers

    in_q = SimpleQueue()
    out_q = SimpleQueue()
    update = {"total": 0}
    semaphore = threading.BoundedSemaphore(maxsize)
    threads: List[MyThread] = []

    t_read = MyThread(
        target=_read_queue_update_total_semaphore,
        args=(it, in_q, semaphore, update, num_workers, progress),
    )
    t_read.start()
    threads.append(t_read)
    for _ in range(num_workers):
        t = MyThread(target=_map_queue, args=(func, in_q, out_q))
        t.start()
        threads.append(t)

    _excepthook = threading.excepthook

    def excepthook(args: threading.ExceptHookArgs):
        exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
        logging.debug("Thread %s interrupted", args.thread, exc_info=exc_info)
        if not isinstance(args.exc_value, KeyboardInterrupt):
            _excepthook(args)

    threading.excepthook = excepthook

    try:
        yield from _read_out_queue_semaphore(out_q, update, semaphore, num_workers, progress)
    except KeyboardInterrupt:
        for thread in threads:
            thread.raise_exc(KeyboardInterrupt)
        for thread in threads:
            thread.join()
        raise
    finally:
        threading.excepthook = _excepthook


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


def executor_ordered(
    func: Callable[[S], T],
    it: Iterable[S],
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
) -> Iterator[Result[T]]:
    q = Queue(maxsize)
    with ThreadPoolExecutor(num_workers) as ex:
        threading.Thread(target=_submit_from_queue, args=(func, it, ex, q)).start()
        yield from progress.track(_queue_reader(q))


def result_wrapper(func: Callable):
    @wraps(func)
    def inner(*args, **kwargs):
        return Result.from_func(func, *args, **kwargs)

    return inner


def parallel_map_thread_unordered(
    func: Callable, it: Iterable, maxsize: int, num_workers: int, progress: ProgressT
) -> Iterator:
    yield from progress.track(
        parallel_map(
            result_wrapper(func),
            it,
            poolcls=ThreadPool,
            ordered=False,
            parallel=True,
            bufsize=maxsize,
            workers=num_workers,
        )
    )


def parallel_map_thread_ordered(
    func: Callable, it: Iterable, maxsize: int, num_workers: int, progress: ProgressT
) -> Iterator:
    yield from progress.track(
        parallel_map(
            result_wrapper(func),
            it,
            poolcls=ThreadPool,
            ordered=True,
            parallel=True,
            bufsize=maxsize,
            workers=num_workers,
        )
    )


def executor_map_thread_unordered(
    func: Callable, it: Iterable, maxsize: int, num_workers: int, progress: ProgressT
) -> Iterator:
    for f in progress.track(
        executor_map(
            func,
            it,
            executercls=ThreadPoolExecutor,
            ordered=False,
            parallel=True,
            bufsize=maxsize,
            workers=num_workers,
        )
    ):
        yield Result.from_finished_future(f)


def executor_map_thread_ordered(
    func: Callable, it: Iterable, maxsize: int, num_workers: int, progress: ProgressT
) -> Iterator:
    for f in progress.track(
        executor_map(
            func,
            it,
            executercls=ThreadPoolExecutor,
            ordered=True,
            parallel=True,
            bufsize=maxsize,
            workers=num_workers,
        )
    ):
        yield Result.from_future(f)


def main():
    TOTAL = 200000
    BUFSIZE = 1000
    NUM_WORKERS = 8

    with RichProgress() as progress:
        p = Progress(progress)

        print("ordered", "map")
        with PrintStatementTime():
            print(list(p.track(map(identity, range(TOTAL))))[:20])

        for func in [
            process_2,
            process_3,
            process,
            parallel_map_thread_unordered,
            # executor_map_thread_unordered,
        ]:
            print("unordered", "thread", func.__name__)
            with PrintStatementTime():
                print(list(func(identity, range(TOTAL), BUFSIZE, NUM_WORKERS, p))[:20])

        for func in [
            executor_ordered,
            parallel_map_thread_ordered,
            # executor_map_thread_ordered,
        ]:
            print("ordered", "thread", func.__name__)
            with PrintStatementTime():
                print(list(func(identity, range(TOTAL), BUFSIZE, NUM_WORKERS, p))[:20])


if __name__ == "__main__":
    main()
