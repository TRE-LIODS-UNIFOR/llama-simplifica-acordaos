import sys
import os
from threading import Lock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.thread_with_result import thread_with_result

def test_thread_with_result():
    results = {
        0: None,
        1: None,
        2: None,
    }

    def func(a, b):
        return a + b

    lock = Lock()

    thread_with_result(func, results, 0, lock, 1, 2)()
    thread_with_result(func, results, 1, lock, 2, 4)()
    thread_with_result(func, results, 2, lock, 3, 4)()
    assert results[0] == 3
    assert results[1] == 6
    assert results[2] == 7
