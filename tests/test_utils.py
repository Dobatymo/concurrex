from genutility.test import MyTestCase

from concurrex.utils import Result


class UtilsTest(MyTestCase):
    def test_create(self):
        truth = "asd"
        result = Result(result=truth).get()
        self.assertEqual(truth, result)

        with self.assertRaises(RuntimeError):
            Result(exception=RuntimeError()).get()

        with self.assertRaises(ValueError):
            Result()

    def test_create_from_func(self):
        truth = "asd"

        def func_result():
            return truth

        def func_exception():
            raise RuntimeError()

        result = Result.from_func(func_result).get()
        self.assertEqual(truth, result)

        with self.assertRaises(RuntimeError):
            Result.from_func(func_exception).get()
