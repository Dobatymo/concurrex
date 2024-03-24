import logging

from genutility.func import identity
from genutility.rich import Progress
from genutility.time import PrintStatementTime
from rich.progress import Progress as RichProgress

from concurrex.thread import (
    _map_unordered_exe,
    _map_unordered_sem,
    _map_unordered_tp,
    executor_ordered,
    parallel_map_thread_ordered,
    parallel_map_thread_unordered,
)


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
            # _map_unordered_bq,
            _map_unordered_sem,
            _map_unordered_tp,
            _map_unordered_exe,
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
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])
    record_factory = logging.getLogRecordFactory()

    def clear_exc_text(*args, **kwargs):
        record = record_factory(*args, **kwargs)
        record.exc_info = None
        return record

    logging.setLogRecordFactory(clear_exc_text)

    main()
