# concurrex (concurrent execution)

The helpers in the Python standard library for concurrent execution, like `threading`, `multiprocessing`, `concurrent.futures` are quite lacking.

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
