# Copyright (c) OpenMMLab. All rights reserved.
from types import MethodType
import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

def import_module_error_func(module_name):
    """When a function is imported incorrectly due to a missing module, raise
    an import error when the function is called."""

    def decorate(func):

        def new_func(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {func.__name__}. '
                'For OpenMMLAB codebases, you may need to install mmcv-full '
                'first before you install the particular codebase. ')

        return new_func

    return decorate


def import_module_error_class(module_name):
    """When a class is imported incorrectly due to a missing module, raise an
    import error when the class is instantiated."""

    def decorate(cls):

        def import_error_init(*args, **kwargs):
            raise ImportError(
                f'Please install {module_name} to use {cls.__name__}. '
                'For OpenMMLAB codebases, you may need to install mmcv-full '
                'first before you install the particular codebase. ')

        cls.__init__ = MethodType(import_error_init, cls)
        return cls

    return decorate
