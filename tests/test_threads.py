import signal
import threading
import traceback
from functools import partial
from itertools import islice

from genutility.callbacks import Progress
from genutility.func import identity
from genutility.test import MyTestCase, parametrize, repeat

from concurrex import imap_unordered_mt


def interrupt_after(seconds: float) -> threading.Timer:
    t = threading.Timer(seconds, partial(signal.raise_signal, signal.SIGINT))
    t.start()
    return t


def find_name_in_exception_traceback(e: BaseException, name: str) -> bool:
    for frame, lineno in traceback.walk_tb(e.__traceback__):
        if name == frame.f_code.co_name:
            return True
    return False


class ThreadTest(MyTestCase):
    @classmethod
    def setUpClass(cls):
        cls.progress = Progress()

    @parametrize(
        ([],),
        (range(1),),
        (range(10),),
        (range(100),),
        (range(1000),),
        (range(10000),),
    )
    def test_lists(self, seq):
        bufsize = 10
        num_workers = 4
        truth = list(seq)
        result = [result.get() for result in imap_unordered_mt(identity, truth, bufsize, num_workers, self.progress)]
        self.assertUnorderedSeqEqual(truth, result)

    @parametrize(
        ([],),
        (range(1),),
        (range(10),),
        (range(100),),
        (range(1000),),
        (range(10000),),
    )
    def test_range(self, it):
        bufsize = 10
        num_workers = 4
        truth = list(it)
        result = [result.get() for result in imap_unordered_mt(identity, it, bufsize, num_workers, self.progress)]
        self.assertUnorderedSeqEqual(truth, result)

    @parametrize(
        (1, 100),
        (10, 100),
        (20, 100),
        (100, 100),
    )
    def test_limited(self, limit, total):
        bufsize = 10
        num_workers = 4
        it = (i for i in range(total))
        gen = imap_unordered_mt(identity, it, bufsize, num_workers, self.progress)
        result = [result.get() for result in islice(gen, limit)]
        remaining_it = list(it)
        remaining_gen = [result.get() for result in gen]

        actual_limit = min(max(bufsize, limit) + 1, total)
        self.assertUnorderedSeqEqual(list(range(limit)), result)
        self.assertEqual(list(range(actual_limit, total)), remaining_it)
        self.assertUnorderedSeqEqual(list(range(limit, actual_limit)), remaining_gen)

    @repeat(20)
    def test_sigint(self):
        bufsize = 10
        num_workers = 4
        it = range(100000)

        interrupt_after(0.1)
        try:
            for result in imap_unordered_mt(identity, it, bufsize, num_workers, self.progress):
                result.get()
        except KeyboardInterrupt as e:
            self.assertTrue(find_name_in_exception_traceback(e, "_read_out_queue_semaphore"))
        else:
            self.fail("call wasn't interrupted")
