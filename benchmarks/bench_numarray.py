import threading

from atomicarray import ArrayInt32
from genutility.time import PrintStatementTime

from concurrex.utils import NumArrayAtomics, NumArrayAtomicsInt, NumArrayPython


def bench(cls_in, cls_add, num_threads=8):
    numarray = cls_in(0, 0, 0)

    def func(numarray):
        for _i in range(100000):
            numarray += cls_add(1, 2, 3)

    threads = [threading.Thread(target=func, args=(numarray,)) for t in range(num_threads)]
    with PrintStatementTime(f"{cls_in.__name__}: {{delta}}"):
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print(list(numarray))


def main():
    bench(ArrayInt32, ArrayInt32)
    bench(NumArrayPython, NumArrayPython)
    bench(NumArrayAtomics, NumArrayAtomicsInt)


if __name__ == "__main__":
    main()
