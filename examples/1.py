import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from genutility.args import is_dir
from rich import print
from rich.prompt import Prompt

from concurrex.thread import PeriodicExecutor, ThreadPool
from concurrex.utils import CvWindow


def decode_image(path: Path) -> Tuple[str, np.ndarray]:
    _path = os.fspath(path)
    img = cv2.imread(_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    return _path, img


def main(basepath: Path) -> None:
    with ThreadPool(num_workers=3) as tp:
        with PeriodicExecutor(lambda: print(tp.num_tasks())), CvWindow() as window:
            for result in tp.map_unordered(decode_image, basepath.rglob("*.jpg"), bufsize=10):
                path, img = result.get()
                window.show(img, path)
                label = Prompt.ask("label:")
                print(path, label)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=is_dir)
    args = parser.parse_args()

    main(args.path)
