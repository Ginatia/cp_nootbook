[DP on Trees Tutorial](https://codeforces.com/blog/entry/20935)

### Problem 1

N个点的树，每个节点 $i$ 有权值 $C_i$  

你可以选择一些节点 ，并保持节点两两不相邻，计算其总和, 求出最大总和    

$max:{ \sum c_i }$

设当前节点为 u , u 的父亲为 f,u相邻的节点为v  

- 若选择 u , 那么不能选择 v ,只能选择v的儿子  

- 若不选择 u , 那么可以选择v,也可以选择v的儿子  

设 dp[u][0]: 不选择 u节点  ,dp[u][1]:选择u节点  

```
dp[u][0]+= std::max(dp[v][0],dp[v][1])  

dp[u][1]+= C[u]+ dp[v][0]  

```

```cpp
//adjacency list
//adj[i] contains all neighbors of i
vector<int> adj[N];

//functions as defined above
int dp1[N],dp2[N];

//pV is parent of node V
void dfs(int V, int pV){

    //for storing sums of dp1 and max(dp1, dp2) for all children of V
    int sum1=0, sum2=0;

    //traverse over all children
    for(auto v: adj[V]){
    if(v == pV) continue;
    dfs(v, V);
    sum1 += dp2[v];
    sum2 += max(dp1[v], dp2[v]);
    }

    dp1[V] = C[V] + sum1;
    dp2[V] = sum2;
}

int main(){
    int n;
    cin >> n;

    for(int i=1; i<n; i++){
    cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
    }

    dfs(1, 0);
    int ans = max(dp1[1], dp2[1]);
    cout << ans << endl;
}

```


### Problem 2

N个点的树，计算任意两个节点之间的最长路径  

设根节点为 root=1  

- f(x) : 最长路径从节点x开始并终止于x的子树节点  
- g(x) : 最长路径从节点x的子树节点开始，经过x，终止于x的子树节点  

答案为   $x \in V, max\{  f(x),g(x) \} $  

if x is leaf,then f[x]=0

$ f(x)=1+max \{f(y)\}$ y is son of x   


$ g(x) = 2+ sum \ of \ two \  max \ \{ f(y) \} $  y is son of x

模板题:
[Tree Diameter](https://cses.fi/problemset/task/1131)

```cpp
void solve()
{
  int n;std::cin>>n;
  std::vector<std::vector<int>>adj(n);

  for(int i=1;i<n;i++){
    int u,v;std::cin>>u>>v;--u,--v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  
  std::vector<int>F(n),G(n);
  int ans=0;

  auto dfs=[&](auto self,int u,int f)->void{
    std::vector<int>val;
    for(auto v:adj[u]){
      if(v==f)continue;
      self(self,v,u);
      val.push_back(F[v]);
    }
    std::sort(val.begin(),val.end(),std::greater<int>());
    if(!val.empty())F[u]=1+val[0];

    if(val.size()>=2){
      G[u]=2+val[0]+val[1];
    }
    ans=std::max({ans,F[u],G[u]});
  };

  dfs(dfs,0,0);
  std::cout<<ans<<'\n';
}
```


