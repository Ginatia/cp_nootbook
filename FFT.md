```python
import numpy as np


def convolve(f, g):
    """
    f:np.ndarray(int64)
    g:np.ndarray(int64)

    return h:np.ndarray,f*g
    """
    fft_len = 1
    while 2*fft_len < len(f)+len(g)-1:
        fft_len *= 2
    fft_len *= 2

    Ff = np.fft.rfft(f, fft_len)
    Fg = np.fft.rfft(g, fft_len)

    Fh = Ff*Fg

    h = np.fft.irfft(Fh, fft_len)

    h = np.rint(h).astype(np.int64)

    return h[:len(f)+len(g)-1]


def convolve_mod(f, g, p):
    """
    f=2**15*f1+f2
    g=2**15*g1+g2
    fg=2**30*f1g1+2**15(f1g2+f2g1)+f2g2
    a=f1g1
    b=f1g2+f2g1
    c=f2g2
    fg=a<<30+b<<15+c
    """
    f1, f2 = np.divmod(f, 1 << 15)
    g1, g2 = np.divmod(g, 1 << 15)

    a = convolve(f1, g1) % p
    c = convolve(f2, g2) % p
    b = (convolve(f1, g2) % p+convolve(f2, g1) % p) % p

    h = (a << 30)+(b << 15)+c
    return h % p

```

[C - 高速フーリエ変換](https://atcoder.jp/contests/atc001/tasks/fft_c)

有N种主菜与副菜，主菜 $A_i$ 元，副菜 $B_i$ 元  
从主菜/副菜 中各选一个，组成套餐的价格为 $A_i+B_j$  
给定价格 $K$ , 有多少种组成价格K的套餐?  
即 $A_i+B_j=K$ 有多少对 $(i,j)$  

$$
f(x)=\sum_{i=0}^{N}A_ix^i \\
g(x)=\sum_{j=0}^{N}B_jx^j \\
(f*g)(x)=\sum_{k=0}^{2N}(\sum_{i=0}^{k}A_iB_{k-i})x^k = \sum_{k=0}^{2N}C_kx^k  \\
$$

```python
def solve():
    n=I()*2
    A=[0]*n
    B=[0]*n
    for i in range(n//2):
        A[i+1],B[i+1]=LI()
    A=np.array(A)
    B=np.array(B)
    C=convolve(A,B)
    for K in range(n):
        print(C[K+1])
```

