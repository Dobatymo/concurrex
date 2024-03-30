import signal
import threading
import traceback
from functools import partial
from itertools import islice
from random import random
from time import sleep
from typing import Callable, Generic, Iterable, Iterator, List, Optional, Set, TypeVar
from unittest import expectedFailure

from genutility.callbacks import Progress
from genutility.func import identity
from genutility.test import MyTestCase, parametrize, parametrize_product, parametrize_starproduct, repeat
from genutility.time import MeasureTime

from concurrex._thread import Result, map_unordered_semaphore
from concurrex._thread_pool import ThreadPool, map_unordered_concurrex
from concurrex.thread import ThreadedIterator

T = TypeVar("T")


def interrupt_after(seconds: float) -> threading.Timer:
    t = threading.Timer(seconds, partial(signal.raise_signal, signal.SIGINT))
    t.start()
    return t


def find_name_in_exception_traceback(e: BaseException, name: str) -> bool:
    for frame, _lineno in traceback.walk_tb(e.__traceback__):
        if name == frame.f_code.co_name:
            return True
    return False


class CheckMemory(Generic[T]):
    def __init__(self) -> None:
        self.items: Set[T] = set()

    def __len__(self) -> int:
        return len(self.items)

    def provide(self, it: Iterable[T], *, seconds: float) -> Iterator[T]:
        for item in it:
            sleep(seconds)
            self.items.add(item)
            yield item

    def collect(self, it: Iterable[Result[T]], *, seconds: float) -> List[T]:
        out = []
        for result in it:
            item = result.get()
            self.items.remove(item)
            sleep(seconds)
            out.append(item)
        return out


class Watcher(threading.Thread):
    results: List

    def __init__(
        self,
        func: Callable[[], T],
        period: float = 0.1,
        name: Optional[str] = None,
        daemon: Optional[bool] = None,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self.period = period
        self.func = func
        self.results: List[T] = []
        self.running = True

    def run(self) -> None:
        while self.running:
            self.results.append(self.func())
            sleep(self.period)

    def stop(self) -> None:
        self.running = False


def identity_sleep(x: T, *, seconds: float) -> T:
    sleep(seconds)
    return x


def identity_random_sleep(x: T) -> T:
    sleep(random() / 10)  # nosec
    return x


class ThreadTest(MyTestCase):
    @classmethod
    def setUpClass(cls):
        cls.progress = Progress()

    @parametrize_product(
        [[], range(1), range(10), range(100), range(1000), range(10000)],
        [map_unordered_semaphore, map_unordered_concurrex],
    )
    def test_lists(self, seq, func):
        bufsize = 10
        num_workers = 4
        truth = list(seq)
        result = [result.get() for result in func(identity, truth, bufsize, num_workers, self.progress)]
        self.assertUnorderedSeqEqual(truth, result)

    @parametrize_product(
        [[], range(1), range(10), range(100), range(1000), range(10000)],
        [map_unordered_semaphore, map_unordered_concurrex],
    )
    def test_range(self, it, func):
        bufsize = 10
        num_workers = 4
        truth = list(it)
        result = [result.get() for result in func(identity, it, bufsize, num_workers, self.progress)]
        self.assertUnorderedSeqEqual(truth, result)

    @parametrize_starproduct(
        [[0, 0.1, 10], [0.1, 0, 0], [0, 0, 0], [0.1, 0.1, 0]],
        [[map_unordered_semaphore], [map_unordered_concurrex]],
    )
    def test_provide_collect(self, provide_wait, collect_wait, min_size, func):
        memory = CheckMemory()
        it = memory.provide(range(30), seconds=provide_wait)
        bufsize = 10
        num_workers = 4
        truth = list(range(30))

        thread = Watcher(lambda: len(memory))
        thread.start()
        result = memory.collect(
            func(identity, it, bufsize, num_workers, self.progress),
            seconds=collect_wait,
        )
        self.assertUnorderedSeqEqual(truth, result)
        thread.stop()

        self.assertLessEqual(max(thread.results), bufsize)
        self.assertGreaterEqual(max(thread.results), min_size)

    @parametrize_starproduct(
        [(1, 100), (10, 100), (20, 100), (100, 100)],
        [[map_unordered_semaphore], [map_unordered_concurrex]],
        [[identity], [identity_random_sleep]],
    )
    def test_limited(self, limit, total, func, mapfunc):
        bufsize = 10
        num_workers = 4
        it = (i for i in range(total))
        gen = func(mapfunc, it, bufsize, num_workers, self.progress)
        result = [result.get() for result in islice(gen, limit)]
        remaining_it = list(it)
        remaining_gen = [result.get() for result in gen]

        all_results = result + remaining_it + remaining_gen
        self.assertEqual(limit, len(result))
        self.assertEqual(total, len(result) + len(remaining_it) + len(remaining_gen))
        self.assertUnorderedSeqEqual(range(total), all_results)

    def _call_and_kill(
        self,
        func: Callable,
        mapfunc: Callable,
        it,
        bufsize,
        num_workers,
        seconds: float,
    ) -> float:
        def consume_results():
            for result in func(mapfunc, it, bufsize, num_workers, self.progress):
                result.get()

        interrupt_after(seconds)
        with MeasureTime() as delta:
            try:
                consume_results()
            except KeyboardInterrupt as e:
                if not find_name_in_exception_traceback(e, "_read_out_queue_semaphore"):
                    pass
                    # self.fail("call wasn't interrupted at the right instruction, so cleanup will fail")
            else:
                self.fail("call wasn't interrupted")
        return delta.get()

    @repeat(10000, verbose=True)
    def _test_sigint_identity(self):
        bufsize = 10
        num_workers = 4
        it = range(100000)
        self._call_and_kill(map_unordered_semaphore, identity, it, bufsize, num_workers, 0.1)

    @expectedFailure
    @repeat(10)
    def _test_sigint_sleep(self):
        bufsize = 10
        num_workers = 4
        it = range(100000)
        delta = self._call_and_kill(
            map_unordered_semaphore,
            partial(identity_sleep, seconds=2),
            it,
            bufsize,
            num_workers,
            0.1,
        )
        self.assertLessEqual(delta, 1.0)

    def test_ThreadedIterator_1(self):
        it = range(10)
        truth = list(it)
        result = list(ThreadedIterator(it, 3))

        self.assertEqual(truth, result)

    def test_ThreadedIterator_gen_close(self):
        def genfunc():
            yield from range(10)

        gen = genfunc()
        tit = ThreadedIterator(gen, 3)
        result1 = list(islice(tit, 5))
        tit.close()
        result2 = list(tit)
        remaining = list(gen)
        buffer = [r.get() for r in tit.buffer]

        self.assertEqual([0, 1, 2, 3, 4], result1)
        truths = ([5], [5, 6], [5, 6, 7], [5, 6, 7, 8])
        self.assertIn(result2, truths)
        self.assertEqual([], remaining)
        self.assertEqual([], buffer)

    def test_ThreadedIterator_gen_slice(self):
        def genfunc():
            yield from range(10)

        gen = genfunc()
        tit = ThreadedIterator(gen, 3)
        result = list(islice(tit, 5))
        remaining = list(gen)
        buffer = [r.get() for r in tit.buffer]

        self.assertEqual([0, 1, 2, 3, 4], result)
        truths = [[9], [8, 9], [7, 8, 9], [6, 7, 8, 9]]
        self.assertIn(remaining, truths)
        truths = [[], [5], [5, 6], [5, 6, 7]]
        self.assertIn(buffer, truths)

    @parametrize(
        ([],),
        (range(1),),
        (range(10),),
        (range(100),),
        (range(1000),),
        (range(10000),),
    )
    def test_tp_executor_simple(self, seq):
        bufsize = 0  # must be zero otherwise it will deadlock
        num_workers = 4
        truth = list(seq)
        with ThreadPool(num_workers, self.progress) as tp:
            executor = tp.executor(bufsize)
            for item in truth:
                executor.execute(identity, item)
            result = []
            for r in executor.iter_unordered():
                item = r.get()
                result.append(item)
        self.assertUnorderedSeqEqual(truth, result)

    @parametrize(
        ([],),
        (range(1),),
        (range(10),),
        (range(100),),
        (range(1000),),
        (range(10000),),
    )
    def test_tp_executor_simple_done(self, seq):
        bufsize = 0  # must be zero otherwise it will deadlock
        num_workers = 4
        truth = list(seq)
        with ThreadPool(num_workers, self.progress) as tp:
            executor = tp.executor(bufsize)
            for item in truth:
                executor.execute(identity, item)
            executor.done()
            result = []
            for r in executor.iter_unordered(wait_done=True):
                item = r.get()
                result.append(item)
        self.assertUnorderedSeqEqual(truth, result)

    @parametrize(
        ([1], [1]),
        ([2], [2, 1]),
        ([3], [3, 10, 5, 16, 8, 4, 2, 1]),
        ([4], [4, 2, 1]),
        ([1, 2, 3, 4], [1] + [2, 1] + [3, 10, 5, 16, 8, 4, 2, 1] + [4, 2, 1]),
    )
    def test_tp_executor_collatz(self, seq, truth):
        bufsize = 0  # must be zero otherwise it will deadlock
        num_workers = 4
        with ThreadPool(num_workers, self.progress) as tp:
            executor = tp.executor(bufsize)
            for item in seq:
                executor.execute(identity, item)
            result = []
            for r in executor.iter_unordered():
                item = r.get()
                result.append(item)
                if item > 1:
                    if item % 2 == 0:
                        executor.execute(identity, item // 2)
                    else:
                        executor.execute(identity, 3 * item + 1)
        if len(seq) == 1:
            self.assertEqual(truth, result)
        else:
            self.assertUnorderedSeqEqual(truth, result)
