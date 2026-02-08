# In the name of Allah.

from collections import deque

def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        
        # Simulate doubly linked list with dictionaries
        nxt = {}  # nxt[i] = next element after i
        prv = {}  # prv[i] = previous element before i
        
        # Initialize circular doubly linked list
        for i in range(n):
            nxt[i] = (i + 1) % n
            prv[i] = (i - 1) % n
        
        def c_next(idx):
            """Get circular next index - O(1)"""
            return nxt[idx]
        
        def c_prev(idx):
            """Get circular previous index - O(1)"""
            return prv[idx]
        
        def hole(idx):
            """Check if element at idx is a 'hole' (smaller than both neighbors)"""
            return a[idx] <= min(a[c_next(idx)], a[c_prev(idx)])
        
        # Initialize queue with all holes
        q = deque()
        for i in range(n):
            if hole(i):
                q.append(i)
        
        ans = 0
        mark = [False] * n
        alive_count = n
        
        while alive_count > 1:
            i = q.popleft()
            
            if mark[i]:
                continue
            
            mark[i] = True
            
            # Add the minimum of the two neighbors
            next_idx = c_next(i)
            prev_idx = c_prev(i)
            ans += min(a[next_idx], a[prev_idx])
            
            # Remove current element by updating pointers
            nxt[prev_idx] = next_idx
            prv[next_idx] = prev_idx
            alive_count -= 1
            
            # Check if neighbors became holes
            if alive_count > 1:
                if not mark[next_idx] and hole(next_idx):
                    q.append(next_idx)
                if not mark[prev_idx] and hole(prev_idx):
                    q.append(prev_idx)
        
        print(ans)

