import logging
import threading
from queue import Empty, Queue, SimpleQueue
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Type, TypeVar, Union

from genutility.callbacks import Progress as ProgressT

from .thread_utils import MyThread, SemaphoreT, ThreadingExceptHook, _Done, make_semaphore, threading_excepthook
from .utils import TOTAL_TIMEOUT, Result, debug_join, get_object_id

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")


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


def _read_out_queue_semaphore(
    out_q: "SimpleQueue[Optional[Result[T]]]",
    update: Dict[str, int],
    semaphore: SemaphoreT,
    num_workers: int,
    progress: ProgressT,
    description: str = "processed",
    transient: bool = False,
) -> Iterator[Result[T]]:
    object_id = get_object_id(logger, "_read_out_queue_semaphore")

    with progress.task(total=update["total"], description=description, transient=transient) as task:
        completed = 0
        task.update(completed=completed, **update)
        while True:
            item = out_q.get()
            if item is None:
                num_workers -= 1
                logger.debug("Queue empty. %d workers remaining.", num_workers, extra={"object": object_id})
                if num_workers == 0:
                    break
            else:
                completed += 1
                task.update(completed=completed, **update)
                yield item
                semaphore.release()


def _drain_queue(queue: "SimpleQueue[Union[Type[_Done], S]]") -> int:
    out = 0
    while True:
        try:
            queue.get_nowait()
            out += 1
        except Empty:
            break
    return out


def _read_queue_update_total_semaphore(
    it: Iterable[S],
    in_q: "SimpleQueue[Union[Type[_Done], S]]",
    semaphore: SemaphoreT,
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
    description: str = "reading",
    transient: bool = False,
) -> None:
    object_id = get_object_id(logger, "_read_queue_update_total_semaphore")

    try:
        for item in progress.track(it, description=description, transient=transient):
            semaphore.acquire()  # notifying it allows waiting exceptions to interrupt it
            update["total"] += 1
            in_q.put(item)
    except KeyboardInterrupt:
        num = _drain_queue(in_q)
        logger.debug("Interrupted, drained input queue (%d items)", num, extra={"object": object_id})
    finally:
        for _ in range(num_workers):
            in_q.put(_Done)


def map_unordered_semaphore(
    func: Callable[[S], T],
    it: Iterable,
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
    daemon: Optional[bool] = True,
) -> Iterator[Result[T]]:
    """
    Apply `func` to items from `it` using a thread pool of size `num_workers`.

    This function limits the total number of concurrently handled items, including items currently being processed, to at most
    `maxsize + 1`. The `+1` accounts for a potential in-flight handoff between threads.

    This method uses a semaphore for backpressure. Results are yielded in arbitrary order as they become available.

    Parameters:
    - func: A callable to apply to each item.
    - it: An iterable of input items.
    - maxsize: Maximum number of concurrently handled items.
    - num_workers: Number of worker threads.
    - progress: is updated with the total number of items seen and the number completed.
    - daemon: Whether the threads should be daemonized (default: True).
    """

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
        daemon=daemon,
    )
    t_read.start()
    threads.append(t_read)
    for _ in range(num_workers):
        t = MyThread(target=_map_queue, args=(func, in_q, out_q), daemon=daemon)
        t.start()
        threads.append(t)

    object_id = get_object_id(logger, "map_unordered_semaphore")

    with ThreadingExceptHook(threading_excepthook):
        try:
            yield from _read_out_queue_semaphore(out_q, update, semaphore, num_workers, progress)
        except (KeyboardInterrupt, GeneratorExit) as e:
            logger.debug("Caught %s, trying to clean up", type(e).__name__, extra={"object": object_id})
            t_read.raise_exc(KeyboardInterrupt)
            if not semaphore.notify_all(timeout=10):  # this can deadlock
                logger.debug("Semaphore timed out", extra={"object": object_id})
                raise RuntimeError("deadlock")

            debug_join(threads, TOTAL_TIMEOUT, extra={"object": object_id})

            any_terminated = False
            for thread in threads:
                any_terminated = any_terminated or thread.terminate(1)
            if any_terminated:
                raise RuntimeError("Terminated blocking threads")
            raise
        except BaseException as e:
            logger.error("Caught %s, trying to clean up", type(e).__name__, extra={"object": object_id})
            raise
        finally:
            debug_join(threads, TOTAL_TIMEOUT, extra={"object": object_id})


def _read_out_queue(
    out_q: "Queue[Optional[Result[T]]]",
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
    description: str = "processed",
    transient: bool = False,
) -> Iterator[Result[T]]:
    object_id = get_object_id(logger, "_read_out_queue")

    with progress.task(total=update["total"], description=description, transient=transient) as task:
        completed = 0
        task.update(completed=completed, **update)
        while True:
            item = out_q.get()
            if item is None:
                num_workers -= 1
                logger.debug("Queue empty. %d workers remaining.", num_workers, extra={"object": object_id})
                if num_workers == 0:
                    break
            else:
                completed += 1
                task.update(completed=completed, **update)
                yield item


def _read_queue_update_total(
    it: Iterable[S],
    in_q: "Queue[Union[Type[_Done], S]]",
    update: Dict[str, int],
    num_workers: int,
    progress: ProgressT,
    description: str = "reading",
    transient: bool = False,
) -> None:
    for item in progress.track(it, description=description, transient=transient):
        update["total"] += 1
        in_q.put(item)
    for _ in range(num_workers):
        in_q.put(_Done)


def map_unordered_boundedqueue(
    func: Callable[[S], T],
    it: Iterable[S],
    maxsize: int,
    num_workers: int,
    progress: ProgressT,
    daemon: Optional[bool] = True,
) -> Iterator[Result[T]]:
    """
    Map `func` using a threadpool of size `num_workers`.
    Keeps a maximum of `maxsize` elements in its input queue (e.g. if reading `it` is faster than processing),
    and a maximum of `maxsize` elements in the output queue (e.g. if consuming the map output is slower than processing).
    So the maximum number of objects handled by this function at any moment is `2*maxsize+num_workers+1`. +1 due to handoffs.

    Results are yielded in arbitrary order as they become available.

    Parameters:
    - func: A callable to apply to each item.
    - it: An iterable of input items.
    - maxsize: Maximum number of concurrently handled items.
    - num_workers: Number of worker threads.
    - progress: is updated with the total number of items seen and the number completed.
    - daemon: Whether the threads should be daemonized (default: True).
    """

    in_q: "Queue[Union[Type[_Done], S]]" = Queue(maxsize)
    out_q: "Queue[Optional[Result[T]]]" = Queue(maxsize)
    threads: List[threading.Thread] = []
    update = {"total": 0}
    object_id = get_object_id(logger, "map_unordered_boundedqueue")

    t_read = threading.Thread(
        target=_read_queue_update_total, args=(it, in_q, update, num_workers, progress), daemon=daemon
    )
    t_read.start()
    threads.append(t_read)
    for _ in range(num_workers):
        t = threading.Thread(target=_map_queue, args=(func, in_q, out_q), daemon=daemon)
        t.start()
        threads.append(t)

    yield from _read_out_queue(out_q, update, num_workers, progress)

    debug_join(threads, TOTAL_TIMEOUT, extra={"object": object_id})
