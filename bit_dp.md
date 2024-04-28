[D - 徒競走](https://atcoder.jp/contests/abc041/tasks/abc041_d)

$N$ 兔子竞走，一共有 $N!$ 种比赛结果  $(N\le 16)$  

$M$ 对关系，描述 兔子 $x$ 先于 兔子 $y$ 完成比赛  

求有多少种符合M对关系的比赛结果  

$$
f(S)=
\begin{cases}
    1 \quad (S=\emptyset)\\
    \sum_{v\in X_S}f(S-\{v\}) \quad (otherwise)
\end{cases}
$$

由题目可知，有向边 $u \rightarrow v$ 成立  

反之 $v \rightarrow u$ 不成立  

检查不成立的边  

初始化: $dp[0]=1$  

向集合 $S$ 添加新的顶点  $v$  ,保证 $v$ 不在集合 $S$ 里  

枚举集合已有的顶点 $u$ , 由于 $u$ 出现于 $v$ 之前  

那么，有向边 $v \rightarrow u$ 不成立  

```cpp
using i64 = long long;
void solve()
{
    int N,M;std::cin>>N>>M;
    std::vector g(N,std::vector<bool>(N,false));
    for(int i=0;i<M;i++){
        int u,v;
        std::cin>>u>>v;
        --u,--v;
        g[v][u]=true;
    }
    std::vector<i64>dp(1<<N);
    dp[0]=1;
    for(int S=0;S<(1<<N);S++){
        for(int v=0;v<N;v++){
            if(!(S>>v&1)){
                bool ok=true;
                for(int u=0;u<N;u++){
                    if(S>>u&1){
                        if(g[v][u])ok=false;
                    }
                }
                if(ok)dp[S+(1<<v)]+=dp[S];
            }
        }
    }
    std::cout<<dp[(1<<N)-1]<<'\n';
}

```

[E - Traveling Salesman among Aerial Cities](https://atcoder.jp/contests/abc180/tasks/abc180_e)

有 $N$ 城市，$(N\le 17)$ , 编号为 $1-N$ ，城市 $i$ 位于坐标 $(X_i,Y_i,Z_i)$  

从 $(a,b,c)$ 到 $(p,q,r)$ 的花费为  $|p-a|+|q-b|+max(0,r-c)$  

***求从城市1出发，访问每个城市至少1次，然后返回城市1的最小花费***  

首先预处理每个城市之间的花费,记为 $g$  

每个城市都要至少访问一次 

- 1 访问至少1次  
- 0 访问0次  

$dp[S][u]:顶点集合S，以u结尾的路径最小花费$

最终状态为 $(1<<N)-1 \quad $  $N$ 位全为1  

初始化  $dp[0][0]=0,dp[1<<i][i]=g[0][i]$  

向集合 $S$ 添加顶点 $v$  ，保证 $S$ 中没有顶点 $v$  

枚举已有顶点  $u$ ,保证 $S$ 中存在顶点 $u$  

更新最小值 $chmin(dp[S|1<<v][v],dp[S][u]+g[u][v])$  

```cpp
using i64 = long long;
using T=std::tuple<int,int,int>;
i64 calc(const T&l,const T&r){
    const auto&[lx,ly,lz]=l;
    const auto&[rx,ry,rz]=r;
    return std::abs(lx-rx)+std::abs(ly-ry)+std::max(0,rz-lz);
}
void chmin(i64&x,i64 y){
    x=std::min(x,y);
}
void solve()
{
    int N;std::cin>>N;
    std::vector<T>a(N);
    for(auto&[x,y,z]:a){
        std::cin>>x>>y>>z;
    }
    std::vector g(N,std::vector<i64>(N));
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(i==j)continue;
            g[i][j]=calc(a[i],a[j]);
        }
    }
    const i64 NIL=-1;
    const i64 INF=1e9;
    std::vector dp(1<<N,std::vector(N,INF));
    dp[0][0]=0;
    for(int i=1;i<N;i++){
        dp[1<<i][i]=g[0][i];
    }
    for(int S=0;S<(1<<N);S++){
        for(int u=0;u<N;u++){
            if(~S>>u&1)continue;
            for(int v=0;v<N;v++){
                if(S>>v&1)continue;
                chmin(dp[S|1<<v][v],dp[S][u]+g[u][v]);
            }
        }
    }
    
    std::cout<<dp[(1<<N)-1][0]<<'\n';
}

```

[E - Get Everything](https://atcoder.jp/contests/abc142/tasks/abc142_e)  


$N$ 个宝箱，编号 $1-N$ $\quad N\le 12$ 

共有 $M$ 个钥匙  

第 $i$ 把钥匙售价为 $a_i$  ,可以打开 $b_i$ 个宝箱  

***请问打开所有宝箱的最小费用，若无法打开所有宝箱，输出-1***  

每个宝箱都要打开，那么最终态为 $(1<<N)-1$  

$dp[S][i]:宝箱集合S，枚举到第 i把钥匙$  

由于一把钥匙可以开 $b_i$ 个宝箱，且不限使用  ，那么一把钥匙对应一个宝箱状态，记为 $c_i$  

初始化: $dp[0][0]=0$  

不使用钥匙 $i$ $chmin(dp[S][i+1],dp[S][i])$  

使用钥匙 $i$ $chmin(dp[S|c_i][i+1],dp[S][i]+a_i)$  

```cpp
using i64 = long long;
void chmin(int&x,int y){
    x=std::min(x,y);
}
void solve()
{
    int N,M;std::cin>>N>>M;
    std::vector<int>a(M),c(M);
    for(int i=0;i<M;i++){
        int b;
        std::cin>>a[i]>>b;
        for(int j=0;j<b;j++){
            int x;std::cin>>x;
            --x;
            c[i]|=1<<x;
        }
    }
    const int INF=1e9;
    std::vector dp(1<<N,std::vector<int>(M+1,INF));
    dp[0][0]=0;
    for(int i=0;i<M;i++){
        for(int S=0;S<(1<<N);S++){
            chmin(dp[S][i+1],dp[S][i]);
            chmin(dp[S|c[i]][i+1],dp[S][i]+a[i]);
        }
    }
    std::cout<<(dp[(1<<N)-1][M]==INF?-1:dp[(1<<N)-1][M])<<'\n';
}
```