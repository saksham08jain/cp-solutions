#Factorials,inverses,nPr,nCr
mod=10**9+7
from types import *
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
def precompute_factorials(maxx):
    factorials=[1]*(maxx+1)
    for i in range(1,maxx):
        factorials[i]=i*factorials[i-1]
        factorials[i]%=mod 
    return factorials
factorial=precompute_factorials(2*10**5+5)

def pow(x,n):
    #raises x**n moduluo mod 
    #uses binary exponentiaion for faster processing
    ans=1
    while n>0:
        if n%2==1:
            ans=(ans*x)%mod 
        x=(x*x)%mod 
        n//=2
    return ans

def extended_euclidean(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_euclidean(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y
def gcd(a,b):
    return extended_euclidean(a,b)[0]

def invmod(a,M):
    #returns modular inverse of a under modulo M using extended euclidean algorithm
    g, x, y = extended_euclidean(a, M)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return (x % M + M) % M

def invs(a, m):
    n = len(a)
    if n == 0:
        return []
    b = [0] * n
    v = 1
    for i in range(n):
        b[i] = v
        v = (v * a[i]) % m

    _, x, _ = extended_euclidean(v, m)
    x = (x % m + m) % m

    for i in range(n - 1, -1, -1):
        b[i] = (x * b[i]) % m
        x = (x * a[i]) % m

    return b

inverse_factorial=invs(factorial,mod)

def nPr(n, r):
    if r > n:
        return 0
    return (factorial[n] * inverse_factorial[n - r]) % mod

def binomial_coefficient(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    return factorial[n] * inverse_factorial[k] % mod * inverse_factorial[n - k] % mod


#LIS

#Class Queue

from collections import deque

class Queue:
    def __init__(self):
        self.queue = deque()
    
    def push(self, item):
        """Inserts an item at the end of the queue."""
        self.queue.append(item)
    
    def pop(self):
        """Removes the item from the front of the queue and returns it."""
        if self.is_empty():
            raise IndexError("pop from an empty queue")
        return self.queue.popleft()
    
    def front(self):
        """Returns the item at the front of the queue without removing it."""
        if self.is_empty():
            raise IndexError("front from an empty queue")
        return self.queue[0]
    
    def is_empty(self):
        """Checks if the queue is empty."""
        return len(self.queue) == 0
    
    def size(self):
        """Returns the number of items in the queue."""
        return len(self.queue)


def lengthOfLIS(nums):
    
    n = len(nums)
    ans = []
    ans.append(nums[0])

    for i in range(1, n):
        if nums[i] > ans[-1]:
            
            ans.append(nums[i])
        else:
           
            low = 0
            high = len(ans) - 1
            while low < high:
                mid = low + (high - low) // 2
                if ans[mid] < nums[i]:
                    low = mid + 1
                else:
                    high = mid
            
            ans[low] = nums[i]
    return len(ans)


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.color = 1  # 1 for red, 0 for black
        self.size = 1  # size of subtree rooted at this node

class OrderedSet:
    def __init__(self):
        self.TNULL = Node(0)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.TNULL.size = 0
        self.root = self.TNULL

    def insert(self, key):
        node = Node(key)
        node.left = self.TNULL
        node.right = self.TNULL

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            y.size += 1
            if node.key < x.key:
                x = x.left
            elif node.key > x.key:
                x = x.right
            else:
                # Key already exists, don't insert
                self._size_fix(y)
                return

        node.parent = y
        if y == None:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node

        if node.parent == None:
            node.color = 0
            return

        if node.parent.parent == None:
            return

        self._fix_insert(node)

    def _fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self._right_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self._left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self._left_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self._right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

        x.size = x.left.size + x.right.size + 1
        y.size = y.left.size + y.right.size + 1

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

        x.size = x.left.size + x.right.size + 1
        y.size = y.left.size + y.right.size + 1

    def _size_fix(self, y):
        while y != None:
            y.size -= 1
            y = y.parent

    def order_of_key(self, k):
        return self._order_of_key_helper(self.root, k)

    def _order_of_key_helper(self, node, k):
        if node == self.TNULL:
            return 0
        if k == node.key:
            return node.left.size
        if k < node.key:
            return self._order_of_key_helper(node.left, k)
        return node.left.size + 1 + self._order_of_key_helper(node.right, k)

    def find_by_order(self, k):
        return self._find_by_order_helper(self.root, k)

    def _find_by_order_helper(self, node, k):
        if node == self.TNULL:
            return None
        left_size = node.left.size
        if k == left_size:
            return node.key
        if k < left_size:
            return self._find_by_order_helper(node.left, k)
        return self._find_by_order_helper(node.right, k - left_size - 1)
from heapq import *
class MinHeap:
    def __init__(self,arr=None):
        if arr is None:
            arr=[]
        self.arr=arr
        heapify(self.arr)
    def push(self,ele):
        heappush(self.arr,ele)
    def pop(self):
        return heappop(self.arr)
    def __len__(self):
        return len(self.arr)
    def __str__(self):
        return str(self.arr)

class MaxHeap:
    def __init__(self, arr=None):
        if arr is None:
            arr = []
        self.arr = arr
        # Convert the input list of (key, value) tuples into a max-heap based on the key
        self._build_max_heap()

    def _build_max_heap(self):
        # Negate keys to simulate max-heap behavior
        self.arr = [(-key, value) for key, value in self.arr]
        heapify(self.arr)

    def push(self, key, value):
        # Push a new (key, value) pair onto the heap
        heappush(self.arr, (-key, value))

    def pop(self):
        # Pop and return the value of the max (key, value) pair
        key, value = heappop(self.arr)
        return -key,value

    def peek(self):
        # Peek at the max (key, value) pair without popping
        key, value = self.arr[0]
        return -key,value

    def __len__(self):
        return len(self.arr)

    def __str__(self):
        # Return the heap as a list of (key, value) pairs
        return str([(key, value) for key, value in self.arr])
def dij(S):
    pq=MinHeap()
    pq.push((0,S))
    dist=[float('inf')]*V
    dist[S]=0
    visited=set()
    while(len(pq)!=0):
        cur_dist,cur=pq.pop()
        if cur in visited:
            continue 
        visited.add(cur)
        #update distances to its neighbors 
        for nei,w in adj[cur]:
            if cur_dist+w<dist[nei]:
                dist[nei]=w+cur_dist
                pq.push((dist[nei],nei))
    #print(*dist)
    return dist
@bootstrap
def get_bridges(adj):
    n = len(adj)
    time = 1
    bridges = []
    parents = [-1] * n

    @bootstrap
    def dfs_timer(node, parent, times, lows):
        nonlocal time
        times[node] = time
        lows[node] = time
        for neighbor in adj[node]:
            if times[neighbor] == -1:
                time += 1
                parents[neighbor] = node
                yield dfs_timer(neighbor, node, times, lows)
            if neighbor != parent:
                lows[node] = min(lows[node], lows[neighbor])
        if parent != -1 and lows[node] > times[parent]:  # Correct condition
            bridges.append((parent, node))
        yield

    times = [-1] * n
    lows = [-1] * n

    for i in range(n):
        if times[i] == -1:
            dfs_timer(i, -1, times, lows)

    return bridges
class DSU:
    def __init__ (self,V):
        self.parents=[i for i in range(V)]
        self.sz=[1]*(V+1) #size of subtrees
        self.islands=V
    def find(self,x):
        #with path compression 
        if x==self.parents[x]:
            return x 
        else:
            self.parents[x]=self.find(self.parents[x])
            return self.parents[x]
    def union(self,x,y):
        parx=self.find(x)
        pary=self.find(y)
        if parx==pary:
            return 
        self.islands-=1
        
        if self.sz[parx]>self.sz[pary]:
            self.parents[pary]=parx
            self.sz[parx]+=self.sz[pary]
        else:
            self.parents[parx]=pary 
            self.sz[pary]+=self.sz[parx]
    def get_sizes(self):
        arr = []
        seen = set()
        for i in range(len(self.parents)):
            root = self.find(i)  # Find the root of i
            if root not in seen:
                arr.append(self.sz[root])  # Add size of the component's root
                seen.add(root)  # Mark the root as seen
        return arr
from math import sqrt

class Mo:
    def __init__(self, arr, queries):
        self.arr = arr  # The array on which queries are performed
        self.n = len(arr)  # Size of the array
        self.queries = self.create_queries(queries)  # Create and sort queries
        self.answers = [0] * len(queries)  # To store the answers for each query
        self.block_size = int(sqrt(self.n))+1  # Set block size for Mo's algorithm

        # Sort the queries using the custom comparator
        self.queries.sort()

        # Initialize current pointers for the range
        self.cur_l = 0
        self.cur_r = -1
        
        # TODO: Initialize your data structure here (like frequency counters, etc.)
        # Example for range sum query:
        self.frequencies = [0]*(max(arr)+1)
        self.count_odd=0
    def create_queries(self, queries):
        block_size = int(sqrt(self.n))
        query_objects = [Query(l, r, i, block_size) for i, (l, r) in enumerate(queries)]
        return query_objects

    def add(self, idx):
        value = self.arr[idx]
        self.frequencies[value] += 1
        if self.frequencies[value] % 2 == 1:  # frequency became odd
            self.count_odd += 1
        elif self.frequencies[value] %2== 0:  # frequency became even
            self.count_odd -= 1

    def remove(self, idx):
        value = self.arr[idx]
        self.frequencies[value] -= 1
        if self.frequencies[value] % 2 == 0:  # frequency became even
            self.count_odd -= 1
        
        if self.frequencies[value] % 2 == 1:  # frequency became odd
            self.count_odd += 1
        
    
    def get_answer(self):
        # Example: return the current sum
        return self.count_odd
    
    def process_queries(self):
        # Process each query and adjust cur_l, cur_r as needed
        for q in self.queries:
            if (q.r-q.l+1)%2==1:
                self.answers[q.idx]=1
                continue
            while self.cur_l > q.l:
                self.cur_l -= 1
                self.add(self.cur_l)
            while self.cur_r < q.r:
                self.cur_r += 1
                self.add(self.cur_r)
            while self.cur_l < q.l:
                self.remove(self.cur_l)
                self.cur_l += 1
            while self.cur_r > q.r:
                self.remove(self.cur_r)
                self.cur_r -= 1
            
            # Store the result of the current query
            self.answers[q.idx] = self.get_answer()
        
        return self.answers

class Query:
    def __init__(self, l, r, idx, block_size):
        self.l = l
        self.r = r
        self.idx = idx
        self.block_size = block_size

    # Define the comparator for sorting queries
    def __lt__(self, other):
        # First compare blocks (l // block_size)
        if self.l // self.block_size != other.l // other.block_size:
            return self.l // self.block_size < other.l // other.block_size
        # If in the same block, compare the right endpoint (r)
        return self.r < other.r
def sieve_of_eratosthenes(limit):
    sieve = [True] * (limit + 1)
    sieve[0], sieve[1] = False, False
    primes = []
    
    for num in range(2, limit + 1):
        if sieve[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                sieve[multiple] = False
    
    return primes

def prime_factorize(n):
    factors = {}
    primes = sieve_of_eratosthenes(int(n**0.5) + 1)
    
    for prime in primes:
        if prime * prime > n:
            break
        while n % prime == 0:
            if prime in factors:
                factors[prime] += 1
            else:
                factors[prime] = 1
            n //= prime
    
    if n > 1:
        factors[n] = 1
    
    return factors
import math
class PrimeTable:
    def __init__(self, n:int) -> None:
        self.n = n
        self.primes = []
        self.min_div = [0] * (n+1)
        self.min_div[1] = 1
 
        mu = [0] * (n+1)
        phi = [0] * (n+1)
        mu[1] = 1
        phi[1] = 1
 
        for i in range(2, n+1):
            if not self.min_div[i]:
                self.primes.append(i)
                self.min_div[i] = i
                mu[i] = -1
                phi[i] = i-1
            for p in self.primes:
                if i * p > n: break
                self.min_div[i*p] = p
                if i % p == 0:
                    phi[i*p] = phi[i] * p
                    break
                else:
                    mu[i*p] = -mu[i]
                    phi[i*p] = phi[i] * (p - 1)
 
    def is_prime(self, x:int):
        if x < 2: return False
        if x <= self.n: return self.min_div[x] == x
        for i in range(2, int(math.sqrt(x))+1):
            if x % i == 0: return False
        return True
 
    def prime_factorization(self, x:int):
        for p in range(2, int(math.sqrt(x))+1):
            if x <= self.n: break
            if x % p == 0:
                cnt = 0
                while x % p == 0: cnt += 1; x //= p
                yield p, cnt
        while (1 < x and x <= self.n):
            p, cnt = self.min_div[x], 0
            while x % p == 0: cnt += 1; x //= p
            yield p, cnt
        if x >= self.n and x > 1:
            yield x, 1
 
    def get_factors(self, x:int):
        factors = [1]
        for p, b in self.prime_factorization(x):
            n = len(factors)
            for j in range(1, b+1):
                for d in factors[:n]:
                    factors.append(d * (p ** j))
        return factors
import sys
import random
 
input = sys.stdin.readline
rd = random.randint(10 ** 9, 2 * 10 ** 9)
class BIT():
    #one indexed
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
 
    def update(self, index, val):
       
        while index <= self.n:
            self.tree[index] += val
            index += index & (-index)
 
    def query(self, r):
        res = 0
        while r > 0:
            res += self.tree[r]
            r -= r & (-r)
        return res
 
    
    def query_lr(self, left, right):
        return self.query(right) - self.query(left - 1)
    #not checked
    def lower_bound(self, val):
        sum = 0
        pos = 0
        maxbits = self.n.bit_length()  # Use the bit length of n
        for i in range(maxbits - 1, -1, -1):  # Start from the most significant bit
            next_pos = pos + (1 << i)
            if next_pos <= self.n and sum + self.tree[next_pos] < val:
                sum += self.tree[next_pos]
                pos = next_pos
        return pos + 1  # Return one-indexed position

def create_coordinate_mapping(arr):
    """
    Create coordinate compression and decompression mappings for an array.

    Returns:
    - rank: An array where rank[i] gives the compressed value of arr[i].
    - value_to_rank: A dictionary mapping original values to their compressed ranks.
    - rank_to_value: A dictionary mapping compressed ranks back to original values.
    """
    # Sort unique values in the array
    unique_values = sorted(set(arr))
    
    # Create mappings for compression and decompression
    value_to_rank = {val: i + 1 for i, val in enumerate(unique_values)}
    rank_to_value = {i + 1: val for i, val in enumerate(unique_values)}
    
    # Replace original array values with their compressed ranks
    rank = [value_to_rank[x] for x in arr]
    
    return rank, value_to_rank, rank_to_value
class SegmentTree:
    def __init__(self, size):
        """
        Initialize a Segment Tree with a given size. By default, all values are set to infinity.
        """
        self.size = size
        self.tree = [float('inf')] * (2 * size)

    def build(self, arr):
        """
        Build the Segment Tree from a given array.
        """
        # Copy array elements into the tree's leaf nodes
        for i in range(len(arr)):
            self.tree[self.size + i] = arr[i]
        # Build the tree by computing parents
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, idx, value):
        """
        Update the value at index `idx` in the Segment Tree.
        """
        # Update the leaf node
        idx += self.size
        self.tree[idx] = value
        # Update the parents
        while idx > 1:
            idx //= 2
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])

    def query(self, left, right):
        """
        Query the minimum value in the range [left, right) (0-based indexing).
        """
        left += self.size
        right += self.size
        result = float('inf')
        while left < right:
            if left % 2 == 1:
                result = min(result, self.tree[left])
                left += 1
            if right % 2 == 1:
                right -= 1
                result = min(result, self.tree[right])
            left //= 2
            right //= 2
        return result