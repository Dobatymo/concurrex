import logging


class DefaultExtraFilter(logging.Filter):
    def __init__(self, **defaults):
        super().__init__()
        self.defaults = defaults

    def filter(self, record):
        for key, value in self.defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def pytest_configure():
    logger = logging.getLogger()
    logger.addFilter(DefaultExtraFilter(object="NaO"))
