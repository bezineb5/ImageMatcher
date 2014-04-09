from functools import wraps
from contextlib import contextmanager
import time

# From http://stackoverflow.com/a/20924212/2196993


def timeit(func):
    @wraps(func)
    def newfunc(*args):
        startTime = time.time()
        result = func(*args)
        elapsedTime = time.time() - startTime
        print('Function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return result
    return newfunc


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))
