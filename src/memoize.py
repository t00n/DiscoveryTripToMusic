import collections
import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def tuplize(self, something):
      if isinstance(something, list) or isinstance(something, tuple):
         something = tuple([self.tuplize(elem) for elem in something])
      return something

   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      args = self.tuplize(args)
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         # print("Using memoized %s with args %s" % (self.func.__name__, str(args)))
         return self.cache[args]
      else:
         # print("Memoizing %s with args %s" % (self.func.__name__, str(args)))
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)