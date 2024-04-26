数位dp的dfs模板

```cpp
template<typename T>
T calc(std::string num){
    //calc on num
    int n=num.size();
    // for not visit
    const T NIL=-1;
    //n,is_limit,...(other condition)
    std::vector dp(n,std::vector(2,NIL));


    // after limit,follow some condition
    auto dfs=[&](auto self,int i,bool is_limit)->T{
        // the end of dfs
        if(i==n){
            //return condition
        }
        auto&res=dp[i][is_limit];//[condition]

        // had calc
        if(res!=NIL){
            return res;
        }

        T ans=0;
        int up=is_limit?s[i]-'0':'9';
        // enumerate digit
        for(int d=0;d<=up;d++){
            //next dfs
            ans+=self(self,i+1,is_limit and d==up);
        }
        return res=ans;
    };  
    return dfs(dfs,0,true);
}

```


[D - 禁止された数字](https://atcoder.jp/contests/abc007/tasks/abc007_4)


```cpp

using i64 = long long;

i64 calc(std::string&& s){
  int n=s.size();
  const i64 NIL=-1;

  std::vector dp(n,std::vector(2,std::vector(2,NIL)));

  auto dfs=[&](auto self,int i,bool is_limit,bool forbid)->i64{
    if(i==n){
      return forbid;
    }
    auto&res=dp[i][is_limit][forbid];
    if(res!=NIL){
      return res;
    }
    i64 ans=0;
    int up=is_limit?s[i]-'0':9;
    for(int d=0;d<=up;d++){
      ans+=self(self,i+1,is_limit and d==up,forbid or d==4 or d==9);
    }
    return res=ans;
  };
  return dfs(dfs, 0, true, false);
}

void solve()
{
  i64 A,B;std::cin>>A>>B;
  std::cout << calc(std::to_string(B)) - calc(std::to_string(A - 1)) << '\n';
}
```

[E - Almost Everywhere Zero](https://atcoder.jp/contests/abc154/tasks/abc154_e)

1-N 之间，恰有K个非零数位的数有多少个

```cpp
using i64 = long long;
void solve()
{
  std::string S;int K;std::cin>>S>>K;

  int n=S.size();
  const i64 NIL=-1;
  std::vector dp(n,std::vector(2,std::vector(K+1,NIL)));

  auto dfs=[&](auto self,int i,bool is_limit,int cnt)->i64{
    if(i==n){
      return cnt==K;
    }
    auto&res=dp[i][is_limit][cnt];
    if(res!=NIL)return res;
    i64 ans=0;
    int up=is_limit?S[i]-'0':9;
    for(int d=0;d<=up;d++){
      if(d){
        if(cnt<K)ans+=self(self,i+1,is_limit and d==up,cnt+1);
      }
      else{
        ans+=self(self,i+1,is_limit and d==up,cnt);
      }
    }
    return res=ans;
  };

  std::cout<<dfs(dfs,0,true,0)<<'\n';

}
```