<h1>SlopeTrick</h1>

```cpp
template <typename T>
struct Slope
{
    const T INF = std::numeric_limits<T>::max() / 3;

    T min_f;
    std::priority_queue<T, std::vector<T>, std::less<>> L;
    std::priority_queue<T, std::vector<T>, std::greater<>> R;
    T add_L, add_R;

private:
    void push_R(const T &a)
    {
        R.push(a - add_R);
    }
    T top_R() const
    {
        if (R.empty())
        {
            return INF;
        }
        else
        {
            return R.top() + add_R;
        }
    }
    T pop_R()
    {
        T val = top_R();
        if (not R.empty())
        {
            R.pop();
        }
        return val;
    }

    void push_L(const T &a)
    {
        L.push(a - add_L);
    }
    T top_L() const
    {
        if (L.empty())
        {
            return -INF;
        }
        else
        {
            return L.top() + add_L;
        }
    }
    T pop_L()
    {
        T val = top_L();
        if (not L.empty())
        {
            L.pop();
        }
        return val;
    }
    size_t size()
    {
        return L.size() + R.size();
    }
    T relu(T x)
    {
        return std::max<T>(0, x);
    }

public:
    Slope() : min_f(0), add_L(0), add_R(0) {}
    // L,R,min_f
    using Q = std::tuple<T, T, T>;

    Q qry() const
    {
        return Q{top_L(), top_R(), min_f};
    }
    // f(x)+=a
    void add_all(const T &a)
    {
        min_f += a;
    }
    // add \_
    // f(x)+=max(a-x,0)
    void add_a_minus_x(const T &a)
    {
        min_f += relu(a - top_R());
        push_R(a);
        push_L(pop_R());
    }
    // add _/
    // f(x)+=max(x-a,0)
    void add_x_minus_a(const T &a)
    {
        min_f += relu(top_L() - a);
        push_L(a);
        push_R(pop_L());
    }
    // add \/
    // f(x)+=|x-a|
    void add_abs(const T &a)
    {
        add_x_minus_a(a);
        add_a_minus_x(a);
    }

    // \/ -> \_
    // f_{new}(x) = min f(y)(y<=x)
    void clear_R()
    {
        while (not R.empty())
        {
            R.pop();
        }
    }

    // \/ -> _/
    // f_{new}(x) = min f(y)(y>=x)
    void clear_L()
    {
        while (not L.empty())
        {
            L.pop();
        }
    }

    // \/ -> \____/
    // f_{new}(x) = min f(y)(x-b<=y<=x-a)
    void shift(const T &a, const T &b)
    {
        assert(a <= b);
        add_L += a;
        add_R += b;
    }

    // \/. -> .\/
    // f_{new}(x) = f(x-a)
    void shift(const T &a)
    {
        shift(a, a);
    }

    T get_val(const T &x)
    {
        T ans = min_f;
        while (not L.empty())
        {
            ans += relu(pop_L() - x);
        }
        while (not R.empty())
        {
            ans += relu(x - pop_R());
        }
        return ans;
    }
    void merge(Slope &s)
    {
        if (s.size() > size())
        {
            std::swap(s.L, L);
            std::swap(s.R, R);
            std::swap(s.add_L, add_L);
            std::swap(s.add_R, add_R);
            std::swap(s.min_f, min_f);
        }
        while (not s.R.empty())
        {
            add_x_minus_a(s.pop_R());
        }
        while (not s.L.empty())
        {
            add_a_minus_x(s.pop_L());
        }
        min_f += s.min_f;
    }
};
```

[F - Absolute Minima](https://atcoder.jp/contests/abc127/tasks/abc127_f)

最初 $f(x)=0$ ,有两种操作:  
1: $g(x)=f(x)+|x-a|+b$  
2: 输出 $min:{f(x)},以及取得最小值f(x)的最小的x$  

```cpp
void solve()
{
    Slope<i64>S;
    int Q;std::cin>>Q;
    while(Q--){
        int o,a,b;
        std::cin>>o;
        if(o==1){
            std::cin>>a>>b;
            S.add_abs(a);
            S.add_all(b);
        }
        else{
            auto[l,r,mf]=S.qry();
            std::cout<<l<<' '<<mf<<'\n';
        }
    }
}
```

[E - 花火](https://atcoder.jp/contests/dwango2016-prelims/tasks/dwango2016qual_e?lang=ja)

烟花的状态为 [ t (时刻) , p (位置) ]  
尽可能减少 烟花同时燃烧时，人所在的坐标与发射烟花的坐标之间的绝对和

$$
min:\sum|p_i-a_i|\\
等价于:\\
dp_i(x)=|x-p_i|+min_{y\le x} dp_{i-1}(y)\\
$$


```cpp
void solve()
{
    Slope<i64>S;
    int N,L;std::cin>>N>>L;
    const int T=1e5;
    std::vector P(T+1,std::vector<int>());
    for(int i=0;i<N;i++){
        int t,p;std::cin>>t>>p;
        P[t].emplace_back(p);
    }
    for(const auto&v:P){
        if(v.empty()){
            continue;
        }
        S.clear_R();
        for(int p:v){
            S.add_abs(p);
        }
    }
    std::cout<<S.min_f<<"\n";
}
```



[C. Sonya and Problem Wihtout a Legend](https://codeforces.com/contest/713/problem/C)

你会得到一个包含 n 个正整数的数组
在一个回合中，您可以选择任何元素并将其增加或减少 1    
目标是通过尽可能少的操作数来严格增加数组  
您可以以任何方式更改元素，它们可以变为负数或等于 0  

严格递增数组可以等价于非递减数组  

$$
a_i<a_{i+1}\\
a_i\le a_{i+1}-1\\
a_i-i\le a_{i+1}-(i+1)\\
$$

所以只需要 $a_i-i$ 即可  
故原命题等价于：
通过尽可能少的操作数(一个元素+,- 1)使得数组为非递减数组  

$dp_i(x)=|x-a_i|+min_{y\le x}dp_{i-1}(y)$

```cpp
void solve()
{
    Slope<i64>S;
    int n;std::cin>>n;
    std::vector<int>a(n);
    for(int i=0;i<n;i++){
        int x;std::cin>>x;
        x-=i;
        a[i]=x;
    }
    for(int x:a){
        S.clear_R();
        S.add_abs(x);
    }
    std::cout<<S.min_f<<'\n';
}

```




