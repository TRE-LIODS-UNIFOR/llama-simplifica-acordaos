def thread_with_result(func, results, key, lock, args):
    def f():
        result = func(*args)
        with lock:
            results[key] = result
    return f
