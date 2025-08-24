 
#HERE ARE PROGRAMMING TIPS 
 
#0. Solve sample test cases BY HAND first only then you are allowed to write code 
#   NEVER WRITE CODE UNTIL YOU ARE SATISFIED BY THE ANALYSIS AND HAVE A PSEUDOCODE 
#1. Recursion equals tree  
#2  Dont Focus on result, focus on problem , dont see standings and dashboard , become zen , FUCK RATING 
#   its YOU and the PROBLEM no one in between 
# 3. Think passively about problems when doing nothing thats the way to grow , anyway YOU dont HAVE ANYTHING BETTER\
#  TO THINK ABOUT 
#4. Dont force a solution on the problem , ask the problem how it wanna be solved 
 
 
#FUCK ANIME , cant really FUCK ANIME LOL
# With a little courage there was a future that could have changed -MUshoku Tensei 
# If i get a little stronger today then theres a future i can protect -Mushoku Tensei 
 
 
 
import sys
import os
import itertools
from math import ceil 
import inspect
from types import *
input = sys.stdin.readline
mod = 10**9 + 7
standard_input, packages, output_together = 1, 1, 0
dfs, hashing, read_from_file = 1, 1, 0
if 1:
 
    if standard_input:
        import io, os, sys
        input = lambda: sys.stdin.readline().strip()
 
        import math
        inf = math.inf
    if packages:
        from io import BytesIO, IOBase
 
        import random
        import os
 
        import bisect
        import typing
        from collections import Counter, defaultdict, deque
        from copy import deepcopy
        from functools import cmp_to_key, lru_cache, reduce
        from heapq import merge, heapify, heappop, heappush, heappushpop, nlargest, nsmallest
        from itertools import accumulate, combinations, permutations, count, product
        from operator import add, iand, ior, itemgetter, mul, xor
        from string import ascii_lowercase, ascii_uppercase, ascii_letters
        from typing import *
        BUFSIZE = 4096
 
    if output_together:
        class FastIO(IOBase):
            newlines = 0
 
            def __init__(self, file):
                self._fd = file.fileno()
                self.buffer = BytesIO()
                self.writable = "x" in file.mode or "r" not in file.mode
                self.write = self.buffer.write if self.writable else None
 
            def read(self):
                while True:
                    b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                    if not b:
                        break
                    ptr = self.buffer.tell()
                    self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
                self.newlines = 0
                return self.buffer.read()
 
            def readline(self):
                while self.newlines == 0:
                    b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                    self.newlines = b.count(b"\n") + (not b)
                    ptr = self.buffer.tell()
                    self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
                self.newlines -= 1
                return self.buffer.readline()
 
            def flush(self):
                if self.writable:
                    os.write(self._fd, self.buffer.getvalue())
                    self.buffer.truncate(0), self.buffer.seek(0)
 
        class IOWrapper(IOBase):
            def __init__(self, file):
                self.buffer = FastIO(file)
                self.flush = self.buffer.flush
                self.writable = self.buffer.writable
                self.write = lambda s: self.buffer.write(s.encode("ascii"))
                self.read = lambda: self.buffer.read().decode("ascii")
                self.readline = lambda: self.buffer.readline().decode("ascii")
 
        sys.stdout = IOWrapper(sys.stdout)
 
    if dfs:
        from types import GeneratorType
 
        def bootstrap(f, stk=[]):
            def wrappedfunc(*args, **kwargs):
                if stk:
                    return f(*args, **kwargs)
                else:
                    to = f(*args, **kwargs)
                    while True:
                        if type(to) is GeneratorType:
                            stk.append(to)
                            to = next(to)
                        else:
                            stk.pop()
                            if not stk:
                                break
                            to = stk[-1].send(to)
                    return to
            return wrappedfunc
 
    if hashing:
        RANDOM = random.getrandbits(20)
        class Wrapper(int):
            def __init__(self, x):
                int.__init__(x)
 
            def __hash__(self):
                return super(Wrapper, self).__hash__() ^ RANDOM
        class MyDict(dict):
            def __setitem__(self, key, value):
                if isinstance(key, int):
                    key = Wrapper(key)
                super().__setitem__(key, value)
            
            def __getitem__(self, key):
                if isinstance(key, int):
                    key = Wrapper(key)
                return super().__getitem__(key)
            
            def __delitem__(self, key):
                if isinstance(key, int):
                    key = Wrapper(key)
                super().__delitem__(key)
            
            def __contains__(self, key):
                if isinstance(key, int):
                    key = Wrapper(key)
                return super().__contains__(key)
    if read_from_file:
        file = open("input.txt", "r").readline().strip()[1:-1]
        fin = open(file, 'r')
        input = lambda: fin.readline().strip()
        output_file = open("output.txt", "w")
        def fprint(*args, **kwargs):
            print(*args, **kwargs, file=output_file)
 # Overwrite print function
def print(*args, sep=' ', end='\n', file=sys.stdout):
    if file is sys.stdout:
        sys.stdout.write(sep.join(map(str, args)) + end)
    else:
        # Fallback for cases where the file argument is used
        builtins.print(*args, sep=sep, end=end, file=file)
def debug(*variable):
    if ONLINE_JUDGE:
        return 
    else:
        print(*variable)
#If you are submitting on cf then i genuinely recommend , asking gpt to convert py recursion 
#to cpp 
#This is for python recusino limit workaround, converts recuirsive function to a generator 
#stack ,just do @bootstap before writing a function
#replace return with yield in your recursive function
def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc
# Functions for input handling
def inp():
    return int(input())
 
def st():
    return input().rstrip('\n')
 
def lis():
    return list(map(int, input().split()))
 
def ma():
    return map(int, input().split())
 
    
def set_online_judge():
    global ONLINE_JUDGE
    ONLINE_JUDGE=True
'''
**********************************************************************************************
CODE STARTS HERE
***************************************************************************************
'''
 
ONLINE_JUDGE = False
set_online_judge()
tcs=inp()
for _ in range(tcs):
    n=inp()
    adj=[[] for i in range(n+1)]
    for i in range(n-1):
        u,v=ma() 
        adj[u].append(v)
        adj[v].append(u)
    ops=float('inf')
    num_leaves=0
    for u in range(1,n+1):
        cur_ops=0
        if(len(adj[u])==1):
            num_leaves+=1
            # if u is a leaf the cur_ops become one but thatd be a discount on all ops 
            # since the invariant is number of leaves 
            cur_ops-=1
        # thats why we multiply with -1 
        for v in adj[u]:
            if len(adj[v])==1:
                # if v is also a leaf 
                cur_ops-=1 
        ops=min(cur_ops,ops)
    print(num_leaves+ops)

            
