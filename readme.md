# concurrex (concurrent execution)

The helpers in the Python standard library for concurrent execution, like `threading`, `multiprocessing`, `concurrent.futures` are lacking for certain use cases like backpressure and flow control.

## Requirements

The library works on Python 3.8+, but a important Python bug (bpo-29988) was fixed in Python 3.10, which makes ctrl-c handling much more reliable.

## Examples

```python

from pillow import Image

def decode_image(path):
	img = Image.open(path)
	img.load()
	return path, img

with ThreadPool(num_workers=3) as tp:

	for result in tp.map(decode_image, Path(".").rglob("*.jpg"), bufsize=10)
		path, img = result.get()
		img.show()
		label = input("label:")
		print(path, label)
```
