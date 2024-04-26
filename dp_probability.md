有1-6等概率的骰子，求掷N次的点数之和不小于K的概率  

对于点数，概率  $ p= \frac{1}{6} $  

那么求得第 $i$ 次 掷骰子，点数之和为 $j,j\le K$ 的 概率  

由于事件是独立的，那么加法即可  

$dp[i+1][min(j+x,K)]+=dp[i][j]/6.0$

```cpp
void solve()
{
    int N,K;std::cin>>N>>K;
    std::vector dp(N+1,std::vector(K+1,0.0));
    dp[0][0]=1.0;
    for(int i=0;i<N;i++){
        for(int j=0;j<=K;j++){
            for(int x=1;x<=6;x++){
                dp[i+1][std::min(K,j+x)]+=dp[i][j]/6.0;
            }
        }
    }
    std::cout<<std::fixed<<std::setprecision(12)<<dp[N][K]<<'\n';
}
```


有一枚硬币，存在正反两面，扔硬币N次，出现至少K次连续正面或者反面的概率  

$P_{正面}=P_{反面}= \frac{1}{2} $  

对于连续出现至少K次的情况有多种，不好统计，因为这种情况可以分段。对于不分段的情况更好维护，也就是连续出现至多K-1次  

转化命题：$P_{至少K次}=1-P_{至多K-1次}$  

$求得，第i次抛硬币，有j个连续一样的概率$  

- 连续一样: $dp[i+1][j+1]+=dp[i][j]/2.0$
- 不一样: $dp[i+1][1]+=dp[i][j]/2.0$  

因为第1次抛硬币一定有1个连续的，所以  

$dp[1][1]=1$

```cpp
void solve()
{
    int N,K;std::cin>>N>>K;
    std::vector dp(N+1,std::vector(K+1,0.0));
    dp[1][1]=1;
    for(int i=1;i<N;i++){
        for(int j=0;j<K;j++){
            if(j+1<K)dp[i+1][j+1]+=dp[i][j]/2.0;
            dp[i+1][1]+=dp[i][j]/2.0;
        }
    }
    double ans=0;
    for(int i=0;i<K;i++){
        ans+=dp[N][i];
    }
    std::cout<<std::fixed<<std::setprecision(12)<<1.0-ans<<'\n';
}

```

[C - トーナメント](https://atcoder.jp/contests/tdpc/tasks/tdpc_tournament)

有 $2^K$人参加K轮的比赛， 每个人有评级 $R_i$,
第 $P$ 个人与第 $Q$ 个人比赛，胜利的概率为: $\frac{1}{1+10^{(R_Q-R_P)/400}}$  

求出每个人最终胜利的概率  

$第i人胜利 \rightarrow 第i人在第K轮后仍然活着$  

dp[i][k] := 第i人在第k轮结束后活下来的概率  

- 最初每个人都活着: dp[i][0]=1.0
- 预先计算第 $i$ 人与第 $j$ 人比赛胜利的可能性为 $P_{i,j}$  
- 记第k轮，还剩下的人的集合为 $S_{i,k}$  
- $dp[i][k]=\sum_{j\in S_{i,k}} dp[i][k-1]\times dp[j][k-1] \times P_{i,j}$  

```cpp
void solve()
{
  int N,K;std::cin>>K;
  N=1<<K;
  std::vector dp(N,std::vector(K+1,0.0));
  std::vector<double>R(N);
  for(auto&x:R)std::cin>>x;
  for(int i=0;i<N;i++)dp[i][0]=1.0;

  std::vector P(N,std::vector(N,0.0));

  auto calc=[&](int i,int j){
    return 1.0/(1.0+std::pow(10.0,(R[j]-R[i])/400.0));
  };

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      P[i][j]=calc(i,j);
    }
  }


  for(int k=1;k<=K;k++){
    for(int i=0;i<N;i++){
      int lb=(i>>k)<<k;
      for(int j=lb;j<lb+(1<<k);j++){
        if((i>>(k-1)&1)==(j>>(k-1)&1))continue;
        dp[i][k]+=dp[i][k-1]*dp[j][k-1]*P[i][j];
      }
    }
  }
  
  std::cout<<std::fixed<<std::setprecision(12);
  for(int i=0;i<N;i++){
    std::cout<<dp[i][K]<<'\n';
  } 
}
```


