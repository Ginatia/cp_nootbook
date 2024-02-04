##### [233. 数字 1 的个数](https://leetcode.cn/problems/number-of-digit-one/)

###### 给定一个整数 `n`，计算所有小于等于 `n` 的非负整数中数字 `1` 出现的个数。

```py
class Solution:
    def countDigitOne(self, n: int) -> int:
        s=str(n)
        @cache
        def f(i:int,cnt:int,is_limit:bool)->int:
            if i==len(s):
                return cnt
            ans=0
            up=int(s[i]) if is_limit else 9
            for d in range(up+1):
                ans+=f(i+1,cnt+(d==1),is_limit and d==up)
            return ans
        return f(0,0,True)
```

##### [2376. 统计特殊整数](https://leetcode.cn/problems/count-special-integers/)

如果一个正整数每一个数位都是 **互不相同** 的，我们称它是 **特殊整数** 。

给你一个 **正** 整数 `n` ，请你返回区间 `[1, n]` 之间特殊整数的数目。



```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)

        @cache
        def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            ans = 0
            if not is_num:
                ans = f(i+1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(1-int(is_num), up+1):
                if mask & (1 << d) == 0:
                    ans += f(i+1, mask | (1 << d), is_limit and d == up, True)
            return ans

        return f(0, 0, True, False)
```

##### [2719. 统计整数数目](https://leetcode.cn/problems/count-of-integers/)

给你两个数字字符串 `num1` 和 `num2` ，以及两个整数 `max_sum` 和 `min_sum` 。如果一个整数 `x` 满足以下条件，我们称它是一个好整数：

* `num1 <= x <= num2`
* `min_sum <= digit_sum(x) <= max_sum`.

请你返回好整数的数目。答案可能很大，请返回答案对 `10**9 + 7` 取余后的结果。

```python
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        def calc(n: str) -> int:
            @cache
            def dfs(i: int, s: int, is_limit: bool) -> int:
                if s > max_sum:
                    return 0
                if i == len(n):
                    return s >= min_sum
                ans = 0
                up = int(n[i]) if is_limit else 9
                for d in range(up+1):
                    ans += dfs(i+1, s+d, is_limit and d == up)
                return ans
            return dfs(0, 0, True)

        is_good = min_sum <= sum(map(int, num1)) <= max_sum
        ans = (calc(num2)-calc(num1)+is_good) % (10**9+7)
        return ans
```

##### [788. 旋转数字](https://leetcode.cn/problems/rotated-digits/)

我们称一个数 X 为好数, 如果它的每位数字逐个地被旋转 180 度后，我们仍可以得到一个有效的，且和 X 不同的数。要求每位数字都要被旋转。

现在我们有一个正整数 `N`, 计算从 `1` 到 `N` 中有多少个数 X 是好数？

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        dif = (0, 0, 1, -1, -1, 1, 1, -1, 0, 1)
        s = str(n)

        @cache
        def dfs(i: int, hasDif: bool, is_limit: bool) -> int:
            if i == len(s):
                return hasDif
            ans = 0
            up = int(s[i]) if is_limit else 9
            for d in range(up+1):
                if dif[d] != -1:
                    ans += dfs(i+1, hasDif or dif[d], is_limit and d == up)
            return ans
        return dfs(0, False, True)
```

##### [902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)

###### 

给定一个按 **非递减顺序** 排列的数字数组 `digits` 。你可以用任意次数 `digits[i]` 来写的数字。例如，如果 `digits = ['1','3','5']`，我们可以写数字，如 `'13'`, `'551'`, 和 `'1351315'`。

返回 _可以生成的小于或等于给定整数 `n` 的正整数的个数_ 。

```python
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        s = str(n)

        @cache
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            ans = 0
            if not is_num:
                ans += dfs(i+1, False, False)
            up = s[i] if is_limit else '9'

            for d in digits:
                if d > up:
                    break
                ans += dfs(i+1, is_limit and d == up, True)
            return ans
        return dfs(0, True, False)
```

##### [面试题 17.06. 2出现的次数](https://leetcode.cn/problems/number-of-2s-in-range-lcci/)

编写一个方法，计算从 0 到 n (含 n) 中数字 2 出现的次数。

```python
class Solution:
    def numberOf2sInRange(self, n: int) -> int:
        s=str(n)
        @cache
        def dfs(i:int,cnt:int,is_limit:bool)->int:
            if i==len(s):
                return cnt
            ans=0
            up=int(s[i]) if is_limit else 9
            for d in range(up+1):
                ans+=dfs(i+1,cnt+(d==2),is_limit and d==up)
            return ans
        return dfs(0,0,True)
```

##### [600. 不含连续1的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/)

给定一个正整数 `n` ，请你统计在 `[0, n]` 范围的非负整数中，有多少个整数的二进制表示中不存在 **连续的 1**

```python
class Solution:
    def findIntegers(self, n: int) -> int:
        s = bin(n)[2:]

        @cache
        def dfs(i: int, one: bool, is_limit: bool) -> int:
            if i == len(s):
                return 1
            ans = 0
            up = int(s[i]) if is_limit else 1
            ans += dfs(i+1, False, is_limit and 0 == up)
            if not one and up == 1:
                ans += dfs(i+1, True, is_limit)
            return ans
        return dfs(0, False, True)
```

##### [E - 数](https://atcoder.jp/contests/tdpc/tasks/tdpc_number#:~:text=%E9%A2%98%E8%A7%A3-,E%20%2D%20%E6%95%B0,-%E9%A2%98%E8%A7%A3)[E - 数](https://atcoder.jp/contests/tdpc/tasks/tdpc_number#:~:text=%E9%A2%98%E8%A7%A3-,E%20%2D%20%E6%95%B0,-%E9%A2%98%E8%A7%A3)

求出[1,n) 有多少个数 满足：数位之和是D的倍数，对答案取模$10^{9}+7$

```cpp
using Z = Mint<i64>;
void solve()
{
    int D;
    std::string s;
    std::cin >> D >> s;
    int n = s.size();
    const i64 NIL = -1;
    std::vector dp(n, std::vector<std::vector<i64>>(D, std::vector<i64>(2, NIL)));

    auto dfs = [&](auto self, int i, int r, bool is_limit) -> Z
    {
        if (i == n)
        {
            return r == 0;
        }
        if (dp[i][r][is_limit] != NIL)
        {
            return dp[i][r][is_limit];
        }
        Z ans = 0;
        int up = is_limit ? int(s[i] - '0') : 9;
        for (int d = 0; d <= up; d++)
        {
            ans += self(self, i + 1, (r + d) % D, is_limit and d == up);
        }
        return dp[i][r][is_limit] = ans.v;
    };
    int cnt = 0;
    for (auto i : s)
    {
        cnt += i - '0';
    }
    bool n_ok = cnt % D == 0;
    std::cout << dfs(dfs, 0, 0, true) - n_ok - 1 << '\n';
}
```

##### [D - 禁止された数字](https://atcoder.jp/contests/abc007/tasks/abc007_4#:~:text=%F0%9F%A7%AA-,D%20%2D%20%E7%A6%81%E6%AD%A2%E3%81%95%E3%82%8C%E3%81%9F%E6%95%B0%E5%AD%97,-%E9%A2%98%E8%A7%A3)

给定范围[A,B],$1\le A < B\le 10^{18}$,求出范围内有多少个数满足：数位中至少有一个4，或有一个6

```cpp
using i64 = long long;

i64 calc(i64 num)
{
    std::string s = std::to_string(num);
    int n = s.size();
    const i64 NIL = -1;
    std::vector dp(n, std::vector<std::vector<i64>>(2, std::vector<i64>(2, NIL)));

    auto dfs = [&](auto self, int i, bool forbid, bool is_limit) -> i64
    {
        if (i == n)
        {
            return forbid;
        }
        auto &res = dp[i][forbid][is_limit];
        if (res != NIL)
        {
            return res;
        }
        i64 ans = 0;
        int up = is_limit ? s[i] - '0' : 9;
        for (int d = 0; d <= up; d++)
        {
            ans += self(self, i + 1, forbid or d == 4 or d == 9, is_limit and d == up);
        }
        return res = ans;
    };
    return dfs(dfs, 0, false, true);
}

void solve()
{
    i64 a, b;
    std::cin >> a >> b;
    std::cout << calc(b) - calc(a - 1) << '\n';
}
```

##### [D - 1](https://atcoder.jp/contests/abc029/tasks/abc029_d#:~:text=%F0%9F%A7%AA-,D%20%2D%201,-%E9%A2%98%E8%A7%A3)

求出[1,n] 的整数，其数的10进制中的1的个数之和

```cpp
using i64 = long long;
i64 calc(i64 num)
{
    std::string s = std::to_string(num);
    int n = s.size();
    const i64 NIL = -1;
    std::vector dp(n, std::vector<std::vector<i64>>(n, std::vector<i64>(2, NIL)));
    auto dfs = [&](auto self, int i, i64 cnt, bool is_limit) -> i64
    {
        if (i == n)
        {
            return cnt;
        }
        auto &ans = dp[i][cnt][is_limit];
        if (ans != NIL)
        {
            return ans;
        }
        i64 res = 0;
        int up = is_limit ? s[i] - '0' : 9;
        for (int d = 0; d <= up; d++)
        {
            res += self(self, i + 1, cnt + (d == 1), is_limit and d == up);
        }
        return ans = res;
    };
    return dfs(dfs, 0, 0, true);
}
void solve()
{
    int n;
    std::cin >> n;
    std::cout << calc(n) << '\n';
}
```

##### [E - Digit Sum Divisible](https://atcoder.jp/contests/abc336/tasks/abc336_e#:~:text=E%20%2D%20Digit%20Sum%20Divisible)

求出[1,n]的符合条件的整数x,x满足x%(x的数位和)==0,

```cpp
using i64 = long long;

i64 calc(i64 num)
{
    std::string s = std::to_string(num);
    int n = s.size();
    i64 ans = 0, D;
    const i64 NIL = -1;
    std::vector<std::vector<std::vector<std::vector<i64>>>> dp;

    auto dfs = [&](auto self, int i, int sd, int r, bool is_limit) -> i64
    {
        if (i == n)
        {
            return sd == D and r == 0;
        }
        auto &ndp = dp[i][sd][r][is_limit];
        if (ndp != NIL)
        {
            return ndp;
        }
        i64 res = 0;
        int up = is_limit ? s[i] - '0' : 9;
        for (int d = 0; d <= up; d++)
        {
            if (sd + d > D)
            {
                continue;
            }
            res += self(self, i + 1, sd + d, (10 * r + d) % D, is_limit and d == up);
        }
        return ndp = res;
    };

    for (D = 1; D <= 14 * 9; D++)
    {
        dp.assign(n, std::vector(D + 1, std::vector(D, std::vector(2, NIL))));
        ans += dfs(dfs, 0, 0, 0, true);
    }
    return ans;
}

void solve()
{
    i64 n;
    std::cin >> n;
    std::cout << calc(n) << '\n';
}
```

###### [Problem - C - Codeforces](https://codeforces.com/contest/1036/problem/C#:~:text=%E7%BF%BB%E8%AF%91-,C.%20Classy%20Numbers,-time%20limit%20per)

求出区间[L,R]中有多少个数满足：其数位中有至多3个非0数位

```cpp
using i64 = long long;
i64 calc(i64 num)
{
    std::string s = std::to_string(num);
    int n = s.size();
    const i64 NIL = -1;
    std::vector dp(n, std::vector(n, std::vector(2, std::vector(2, NIL))));
    auto dfs = [&](auto self, int i, int cnt, bool is_limit, bool is_num) -> i64
    {
        if (i == n)
        {
            return is_num and cnt <= 3;
        }
        auto &ndp = dp[i][cnt][is_limit][is_num];
        if (ndp != NIL)
        {
            return ndp;
        }
        i64 ans = 0;
        int up = is_limit ? s[i] - '0' : 9;
        if (not is_num)
        {
            ans = self(self, i + 1, cnt, false, false);
        }
        for (int d = 0; d <= up; d++)
        {
            if ((not is_num and d != 0) or is_num)
            {
                ans += self(self, i + 1, cnt + (d != 0), is_limit and d == up, true);
            }
        }
        return ndp = ans;
    };
    return dfs(dfs, 0, 0, true, false);
}
void solve()
{
    i64 L, R;
    std::cin >> L >> R;
    std::cout << calc(R) - calc(L - 1) << '\n';
}
```

######[No.741 AscNumber(Easy)](https://yukicoder.me/problems/no/741)
求出[1,$10^{n}$]有多少个数字满足：其数位不递减  
对$10^{9}+7$取模

```cpp
using Z = Mint<i64>;
void solve()
{
    int n;
    std::cin >> n;
    std::string s = "1" + std::string(n, '0');
    n = n + 1;
    const i64 NIL = -1;
    std::vector dp(n, std::vector(10, NIL));
    auto dfs = [&](auto self, int i, int cur, bool is_limit, bool is_asc) -> Z
    {
        if (i == n)
        {
            return is_asc;
        }
        auto &ndp = dp[i][cur];
        if (ndp != NIL)
        {
            return ndp;
        }
        Z ans = 0;
        int up = is_limit ? s[i] - '0' : 9;
        for (int d = 0; d <= up; d++)
        {
            if (d >= cur)
            {
                ans += self(self, i + 1, d, is_limit and d == up, true);
            }
        }
        return ndp = ans.v;
    };
    std::cout << dfs(dfs, 0, 0, true, true) << '\n';
}
```

######[D. Beautiful numbers](https://codeforces.com/problemset/problem/55/D)
给定范围[L,R],  
当且仅当正整数可以被其非零位数整除时，它才是美丽的.计算给定范围内美丽数字的数量。

**多组重复贡献时，贴上限,is_limit==true的位不进行记忆化**

```cpp
#include <bits/stdc++.h>
#include <bits/extc++.h>
using i64 = long long;
const int N = 20;
const int M = 2520;
const i64 NIL = -1;
int id[M + 1];
void init()
{
    for (int i = 1, j = 1; i <= M; i++)
    {
        if (M % i == 0)
        {
            id[i] = j++;
        }
    }
}
std::vector<std::vector<std::vector<std::vector<i64>>>> dp;
i64 calc(i64 num)
{
    std::string s = std::to_string(num);
    std::reverse(s.begin(), s.end());
    int n = s.size();

    auto dfs = [&](auto self, int i, int cur, int r, bool is_limit) -> i64
    {
        if (i == -1)
        {
            return r % cur == 0;
        }
        auto &ndp = dp[i][id[cur]][r][is_limit];
        if (not is_limit and ndp != NIL)
        {
            return ndp;
        }
        i64 res = 0;
        int up = is_limit ? s[i] - '0' : 9;
        for (int d = 0; d <= up; d++)
        {
            int ncur = d ? cur / std::gcd(cur, d) * d : cur;
            int nr = (r * 10 + d) % M;
            res += self(self, i - 1, ncur, nr, is_limit and d == up);
        }
        if (not is_limit)
        {
            ndp = res;
        }
        return res;
    };
    return dfs(dfs, n - 1, 1, 0, true);
}
void solve()
{
    i64 L, R;
    std::cin >> L >> R;
    std::cout << calc(R) - calc(L - 1) << '\n';
    // std::cout << calc(R) << ' ' << calc(L - 1) << '\n';
}
int main()
{
    std::cin.tie(nullptr)->sync_with_stdio(false);

    init();
    dp = std::vector(N, std::vector(60, std::vector(M, std::vector(2, NIL))));

    int t;
    std::cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}
```
