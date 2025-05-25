import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future, _AsCompletedWaiter
from concurrent.futures.thread import BrokenThreadPool, _shutdown, _WorkItem
from queue import Queue
from typing import Callable, Iterable, Iterator, Optional, Set, Type, TypeVar, Union

from .thread_utils import _Done
from .utils import Result, with_progress

T = TypeVar("T")
S = TypeVar("S")


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


def _queue_reader(q: "Queue[Optional[Future[T]]]") -> Iterator[Result[T]]:
    """requires active executor"""

    while True:
        item = q.get()
        if item is None:
            break
        yield Result.from_future(item)


@with_progress
def map_ordered_executor(
    func: Callable[[S], T],
    it: Iterable[S],
    maxsize: int,
    num_workers: int,
) -> Iterator[Result[T]]:
    q: "Queue[Optional[Future[T]]]" = Queue(maxsize)
    with ThreadPoolExecutor(num_workers) as ex:
        threading.Thread(target=_submit_from_queue, args=(func, it, ex, q)).start()
        yield from _queue_reader(q)


class ThreadPoolExecutorWithFuture(ThreadPoolExecutor):
    def submit(self, f, fn, /, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _shutdown:
                raise RuntimeError("cannot schedule new futures after interpreter shutdown")

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
    """Process queue using ThreadPoolExecutor"""

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
def map_unordered_executor_in_thread(
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
