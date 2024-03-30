import logging

from genutility.func import identity
from genutility.rich import Progress
from genutility.time import PrintStatementTime
from rich.progress import Progress as RichProgress

from concurrex._thread import map_unordered_boundedqueue, map_unordered_semaphore
from concurrex._thread_concurrent import (
    map_ordered_executor_map,
    map_ordered_parallel_map,
    map_unordered_executor_map,
    map_unordered_parallel_map,
)
from concurrex._thread_exe import map_ordered_executor, map_unordered_executor_in_thread
from concurrex._thread_pool import map_unordered_concurrex


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
            map_unordered_semaphore,
            map_unordered_concurrex,
            map_unordered_executor_in_thread,
            map_unordered_parallel_map,
            map_unordered_boundedqueue,
            map_unordered_executor_map,
        ]:
            print("unordered", "thread", func.__name__)
            with PrintStatementTime():
                print(list(func(identity, range(TOTAL), BUFSIZE, NUM_WORKERS, p))[:20])

        for func in [
            map_ordered_executor,
            map_ordered_parallel_map,
            map_ordered_executor_map,
        ]:
            print("ordered", "thread", func.__name__)
            with PrintStatementTime():
                print(list(func(identity, range(TOTAL), BUFSIZE, NUM_WORKERS, p))[:20])


if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])
    record_factory = logging.getLogRecordFactory()

    def clear_exc_text(*args, **kwargs):
        record = record_factory(*args, **kwargs)
        record.exc_info = None
        return record

    logging.setLogRecordFactory(clear_exc_text)

    main()
