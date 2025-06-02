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
from typing_extensions import Self

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
        self._stop = threading.Event()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def run(self) -> None:
        self.results.append(self.func())
        while not self._stop.wait(self.period):
            self.results.append(self.func())

    def stop(self) -> None:
        self._stop.set()


def identity_sleep(x: T, *, seconds: float) -> T:
    sleep(seconds)
    return x


def identity_random_sleep(x: T) -> T:
    sleep(random() / 10)  # nosec
    return x


def range_sleep(stop: int, seconds: float):
    for i in range(stop):
        yield i
        sleep(seconds)


def genrange(stop):
    yield from range(stop)


class ThreadedIteratorTest(MyTestCase):
    """Some of the tests leak resources since the thread is not closed correctly.
    This is done so that closing can be tested seperatly and incorrect closing methods don't bloc
    otherwise unrelated tests.
    The leaked threads are demons which will be clean up after test/process exit.
    """

    def test_basic(self):
        length = 10
        it = range(length)
        truth = list(it)
        tit = ThreadedIterator(it, 3)

        self.assertEqual(truth, list(tit))
        self.assertEqual(length, tit.processed())
        self.assertFalse(tit.thread.is_alive())

    def test_full(self):
        maxsize = 3
        it = range_sleep(10, 0.01)
        tit = ThreadedIterator(it, maxsize)
        tit._wait_for_queue_full()

        self.assertIn(tit.processed(), [maxsize, maxsize - 1])
        self.assertTrue(tit.thread.is_alive())  # leak

    def test_len(self):
        length = 10
        maxsize = 3
        tit = ThreadedIterator(range(length), maxsize)
        result = len(tit)

        self.assertEqual(length, result)
        self.assertTrue(tit.thread.is_alive())  # leak

    def test_len_fail(self):
        length = 10
        tit = ThreadedIterator((i for i in range(length)), 3)
        with self.assertRaises(TypeError):
            len(tit)
        self.assertTrue(tit.thread.is_alive())  # leak

    def test_gen_close(self):
        gen = genrange(10)
        tit = ThreadedIterator(gen, 3)
        result1 = list(islice(tit, 5))
        result_close = [r.get() for r in tit.close()]
        result2 = list(tit)
        remaining = list(gen)
        buffer = [r.get() for r in tit.buffer]

        self.assertEqual([0, 1, 2, 3, 4], result1)
        self.assertIn(result_close, [[], [5], [5, 6], [5, 6, 7]])
        self.assertIn(result2, [[], [5], [5, 6], [5, 6, 7], [5, 6, 7, 8]])
        self.assertEqual([], remaining)  # because it was closed as well
        self.assertEqual([], buffer)  # drained by close()
        self.assertFalse(tit.thread.is_alive())

    def test_context(self):
        gen = genrange(10)
        with ThreadedIterator(gen, 3) as tit:
            result1 = list(islice(tit, 5))

        self.assertEqual([0, 1, 2, 3, 4], result1)

    def test_context_sleep(self):
        gen = genrange(10)
        with ThreadedIterator(gen, 3) as tit:
            result1 = list(islice(tit, 5))
            sleep(0.1)

        self.assertEqual([0, 1, 2, 3, 4], result1)

    def test_gen_slice(self):
        gen = genrange(10)
        tit = ThreadedIterator(gen, 3)
        result = list(islice(tit, 5))
        remaining = list(gen)
        buffer = [r.get() for r in tit.buffer]

        self.assertEqual([0, 1, 2, 3, 4], result)
        self.assertIn(remaining, [[9], [8, 9], [7, 8, 9], [6, 7, 8, 9], [5, 6, 7, 8, 9]])
        self.assertIn(buffer, [[], [5], [5, 6], [5, 6, 7]])
        self.assertTrue(tit.thread.is_alive())  # leak


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

        with Watcher(lambda: len(memory)) as thread:
            result = memory.collect(
                func(identity, it, bufsize, num_workers, self.progress),
                seconds=collect_wait,
            )
            self.assertUnorderedSeqEqual(truth, result)

        if thread.results:
            max_items_processed_simultaneously = max(thread.results)
            self.assertLessEqual(
                max_items_processed_simultaneously, bufsize + 1
            )  # plus 1 to allow for unimportant races
            # self.assertGreaterEqual(max_items_processed_simultaneously, min_size)  # doesn't have to be true although it should
        else:
            self.fail("Watcher didn't catch any data. This is unusual.")

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

    @parametrize(
        (map_unordered_semaphore,),
        (map_unordered_concurrex,),
    )
    @repeat(10, verbose=True)
    def test_sigint_identity(self, func):
        bufsize = 10
        num_workers = 4
        it = range(100000)
        self._call_and_kill(func, identity, it, bufsize, num_workers, 0.1)

    @parametrize(
        (map_unordered_semaphore,),
        (map_unordered_concurrex,),
    )
    @expectedFailure  # sleep cannot be interrupted
    def test_sigint_sleep(self, func):
        bufsize = 10
        num_workers = 4
        it = range(100000)
        delta = self._call_and_kill(
            func,
            partial(identity_sleep, seconds=2),
            it,
            bufsize,
            num_workers,
            0.1,
        )
        self.assertLessEqual(delta, 1.0)

    def test_tp_nop(self):
        num_workers = 4
        with ThreadPool(num_workers, self.progress):
            pass

    @parametrize(
        ([],),
        (range(1),),
        (range(10),),
        (range(100),),
        (range(1000),),
        (range(10000),),
    )
    def test_tp_executor_simple_no_done(self, seq):
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
