from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from multiprocessing.pool import ThreadPool as MultiprocessingThreadPool
from typing import Callable, Iterable, Iterator, TypeVar

from genutility.concurrency import executor_map, parallel_map
from typing_extensions import ParamSpec

from .utils import Result, with_progress

T = TypeVar("T")
P = ParamSpec("P")


def result_wrapper(func: Callable[P, T]) -> Callable[P, Result[T]]:
    @wraps(func)
    def inner(*args: P.args, **kwargs: P.kwargs) -> Result[T]:
        return Result.from_func(func, *args, **kwargs)

    return inner


@with_progress
def map_unordered_executor_map(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator[Result]:
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
def map_ordered_executor_map(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator[Result]:
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


@with_progress
def map_unordered_parallel_map(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator[Result]:
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
def map_ordered_parallel_map(func: Callable, it: Iterable, maxsize: int, num_workers: int) -> Iterator[Result]:
    yield from parallel_map(
        result_wrapper(func),
        it,
        poolcls=MultiprocessingThreadPool,
        ordered=True,
        parallel=True,
        bufsize=maxsize,
        workers=num_workers,
    )
