from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def newfunc(*args):
        startTime = time.time()
        result = func(*args)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return result
    return newfunc
