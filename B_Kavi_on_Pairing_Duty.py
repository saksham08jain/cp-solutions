 
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
"""
OPTIMIZED PRIMETABLE - TIME COMPLEXITY ANALYSIS
========================================================================================
Method                  | Time Complexity          | Space Complexity | Notes
========================================================================================
__init__(n)            | O(n log n)               | O(n)            | Linear sieve + divisor sieve
is_prime(x)            | O(1) or O(√x)           | O(1)            | O(1) if x ≤ n
prime_factorization(x) | O(log x) or O(√x)       | O(log x)        | O(log x) if x ≤ n
get_factors(x)         | O(d(x))                 | O(d(x))         | d(x) = divisor count
num_factors(x)         | O(1) or O(log x)        | O(1)            | O(1) if x ≤ n (sieve lookup)
get_prime_factors(x)   | O(log x) or O(√x)       | O(log x)        | O(log x) if x ≤ n
sum_of_divisors(x)     | O(log x) or O(√x)       | O(1)            | Uses prime factorization
========================================================================================

KEY OPTIMIZATIONS:
1. ✅ Added divisor count sieve - O(1) lookup for num_factors() when x ≤ n
2. ✅ Added get_prime_factors() using SPF for O(log x) factorization
3. ✅ Optimized get_factors() - generates divisors efficiently
4. ✅ Improved is_prime() - 6k±1 optimization for trial division
5. ✅ All arrays computed during initialization

IMPORTANT: The optimized version behaves IDENTICALLY to the original for all methods.
The only change is that num_factors() is now O(1) instead of O(√x) for x ≤ n.
"""

import math

class PrimeTable:
    def __init__(self, n: int) -> None:
        """
        Initialize prime table with linear sieve + divisor counting sieve.
        
        Time Complexity: O(n log n)
        - O(n log log n) for linear sieve
        - O(n log n) for divisor counting sieve
        
        Space Complexity: O(n)
        
        Precomputes:
        - primes: list of all primes ≤ n
        - min_div[i]: smallest prime factor of i (for fast factorization)
        - div_count[i]: number of divisors of i (includes 1 and i)
        - mu[i]: Möbius function μ(i)
        - phi[i]: Euler's totient function φ(i)
        """
        self.n = n
        self.primes = []
        self.min_div = [0] * (n + 1)
        self.min_div[1] = 1
        
        # NEW: Divisor count array for O(1) lookups
        self.div_count = [0] * (n + 1)
 
        mu = [0] * (n + 1)
        phi = [0] * (n + 1)
        mu[1] = 1
        phi[1] = 1
 
        # Phase 1: Linear sieve (Euler's sieve) - O(n log log n)
        for i in range(2, n + 1):
            if not self.min_div[i]:
                self.primes.append(i)
                self.min_div[i] = i
                mu[i] = -1
                phi[i] = i - 1
            for p in self.primes:
                if i * p > n: break
                self.min_div[i * p] = p
                if i % p == 0:
                    phi[i * p] = phi[i] * p
                    break
                else:
                    mu[i * p] = -mu[i]
                    phi[i * p] = phi[i] * (p - 1)
        
        # Phase 2: NEW - Divisor counting sieve - O(n log n)
        # This gives the SAME results as the original num_factors() method
        for i in range(1, n + 1):
            for j in range(i, n + 1, i):
                self.div_count[j] += 1
    
    def is_prime(self, x: int) -> bool:
        """
        Check if x is prime.
        Time: O(1) if x ≤ n, O(√x) otherwise
        Space: O(1)
        
        IDENTICAL to original implementation for x ≤ n.
        Optimized for x > n using 6k±1 optimization.
        """
        if x < 2: return False
        if x <= self.n: return self.min_div[x] == x
        
        # Optimized trial division for x > n
        if x == 2: return True
        if x % 2 == 0: return False
        if x % 3 == 0: return False
        
        # Check only 6k±1 candidates
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def prime_factorization(self, x: int):
        """
        Generate prime factorization as (prime, exponent) pairs.
        Time: O(log x) if x ≤ n, O(√x) otherwise
        Space: O(log x)
        
        IDENTICAL to original implementation - NO CHANGES.
        """
        # Trial division for factors not in table
        for p in range(2, int(math.sqrt(x)) + 1):
            if x <= self.n: break
            if x % p == 0:
                cnt = 0
                while x % p == 0: 
                    cnt += 1
                    x //= p
                yield p, cnt
        
        # Use precomputed table for remaining factors
        while (1 < x and x <= self.n):
            p, cnt = self.min_div[x], 0
            while x % p == 0: 
                cnt += 1
                x //= p
            yield p, cnt
        
        # If x is still > 1 and > n, it's a prime
        if x > self.n and x > 1:
            yield x, 1
    
    def get_factors(self, x: int):
        """
        Get all divisors of x.
        Time: O(√x + d(x)²) or O(d(x)) depending on approach
        Space: O(d(x))
        
        IDENTICAL logic to original - just slightly cleaner code.
        Returns the SAME list of factors.
        """
        factors = [1]
        for p, b in self.prime_factorization(x):
            n = len(factors)
            for j in range(1, b + 1):
                for d in factors[:n]:
                    factors.append(d * (p ** j))
        return factors
    
    def num_factors(self, x: int) -> int:
        """
        Count number of divisors of x (including 1 and x).
        
        Time: O(1) if x ≤ n (NEW!), O(log x) otherwise
        Space: O(1)
        
        OPTIMIZATION: For x ≤ n, uses precomputed O(1) lookup instead of
        recalculating via prime factorization.
        
        For x > n, falls back to the EXACT same logic as original.
        
        Formula: If x = p1^a1 * p2^a2 * ... * pk^ak, 
                 then d(x) = (a1+1) * (a2+1) * ... * (ak+1)
        """
        if x <= 0:
            return 0
        if x <= self.n:
            # NEW: O(1) lookup from precomputed sieve
            return self.div_count[x]
        
        # ORIGINAL: For x > n, compute using prime factorization
        ans = 1
        for p, count in self.prime_factorization(x):
            ans *= (count + 1)
        return ans
    
    def get_prime_factors(self, x: int) -> list:
        """
        NEW HELPER: Fast prime factorization using precomputed SPF.
        Returns list of (prime, exponent) tuples.
        Time: O(log x) if x ≤ n, O(√x) otherwise
        Space: O(log x)
        """
        if x <= 1:
            return []
        
        if x <= self.n:
            factors = []
            while x > 1:
                p = self.min_div[x]
                count = 0
                while x % p == 0:
                    count += 1
                    x //= p
                factors.append((p, count))
            return factors
        else:
            return list(self.prime_factorization(x))
    
    def sum_of_divisors(self, x: int) -> int:
        """
        NEW HELPER: Calculate sum of all divisors of x.
        Time: O(log x) if x ≤ n, O(√x) otherwise
        Space: O(1)
        """
        if x <= 0:
            return 0
        if x == 1:
            return 1
        
        result = 1
        for p, exp in self.prime_factorization(x):
            result *= (pow(p, exp + 1) - 1) // (p - 1)
        return result




    
pt=PrimeTable(10**6+5)

mod=998244353
ONLINE_JUDGE = False
set_online_judge()
#if i can make count(x) even log(x) i am done 
#i think it is possible 
#via sieve ig 
def count(x):
    return (pt.num_factors(x))
    # ans=0
    # for i in range(1,x+1):
    #     if x%i==0:
    #         ans+=1
    # return ans

for _ in range(1):
    n=inp()
    dp=[-1]*(n+1)
    dp[1]=1 
    for i in range(2,n+1):
        dp[i]=dp[i-1]*2+count(i)-count(i-1)
        dp[i]%=mod
    print(dp[n])