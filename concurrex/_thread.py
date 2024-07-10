import logging
import threading
from queue import Empty, Queue, SimpleQueue
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Type, TypeVar, Union

from genutility.callbacks import Progress as ProgressT

from .thread_utils import MyThread, SemaphoreT, ThreadingExceptHook, _Done, make_semaphore, threading_excepthook
from .utils import Result

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


def map_unordered_semaphore(
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
        except (KeyboardInterrupt, GeneratorExit) as e:
            logging.debug("Caught %s, trying to clean up", type(e).__name__)
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


def map_unordered_boundedqueue(
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
