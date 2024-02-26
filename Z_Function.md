```python
def z_func(s: str) -> list:
    n = len(s)
    z = [0]*(n)
    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r-i, z[i-l])
        while i+z[i] < n and s[z[i]] == s[i+z[i]]:
            z[i] += 1
        if i+z[i] > r:
            l, r = i, i+z[i]
    return z
```

[3031. 将单词恢复初始状态所需的最短时间 II](https://leetcode.cn/problems/minimum-time-to-revert-word-to-initial-state-ii/description/)

假定我们要移动 $x=t\times k$ 个字符  
那么移动之后剩下的字符串要与原字符串的前缀匹配  
$s[x:]==s[0:(n-x)]$  
否则我们移动掉原字符串  

```python
class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        z=z_func(word)
        n=len(word)
        for i in range(1,n):
            if i%k==0 and z[i]>=n-i:
                return i//k
        return (n+k-1)//k

```

