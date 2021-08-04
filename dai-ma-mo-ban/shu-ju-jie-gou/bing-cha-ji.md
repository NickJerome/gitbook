# 并查集

## 常规写法

```cpp
int fa[N];

void init () {
    for (int i = 1; i <= n; ++ i) fa[i] = i;
}
int find(int x) {
    if (fa[x] == x) return x;
    return fa[x] = find(fa[x]);
}
```

## 一句话写法

```cpp
int p[N];

void init () {
    for (int i = 1; i <= n; ++ i) p[i] = i;
}
int find(int x) {
    return p[x] == x ? x : p[x] = find(p[x]);
}
```

## 带权并查集

> 维护两个点之间的信息，可以是距离，奇偶状态等等

```cpp
int p[N];
void init() {
    for (int i = 1; i <= n; ++ i) p[i] = i;
}
int find(int x){
    if (p[x] == x) return x;
    else {
        int root = find(p[x]);
        // Todo 如：d[x] += d[p[x]];
        return p[x] = root;
    }
}
```

## 并查集例题
https://acm.hdu.edu.cn/showproblem.php?pid=7001
> 可能卡常,用fread读入优化
> 由于查询区间为1到i-1, 所以可以考虑用并查集,在位置x修改为1的时候与后一位合并
> 查询的时候查询1是否存在,如果结果等于x的时候,说明如果此时x位置改成1,i的位置还能往后移动,所以i更新为find(x+1)

```cpp
#include <bits/stdc++.h>
using namespace std;

namespace _{  
    char buf[100000], *p1 = buf, *p2 = buf; bool rEOF = 1;//为0表示文件结尾  
    inline char nc(){ return p1 == p2 && rEOF && (p2 = (p1 = buf) + fread(buf, 1, 100000, stdin), p1 == p2) ? (rEOF = 0, EOF) : *p1++; }  
    template<class _T>  
    inline void read(_T &num){  
        char c = nc(), f = 1; num = 0;  
        while (c<'0' || c>'9')c == '-' && (f = -1), c = nc();  
        while (c >= '0'&&c <= '9')num = num * 10 + c - '0', c = nc();  
        num *= f;  
    }  
    inline bool need(char &c){ return c >= 'a'&&c <= 'z' || c >= '0'&&c <= '9' || c >= 'A'&&c <= 'Z'; }//读入的字符范围  
    inline void read_str(char *a){  
        while ((*a = nc()) && !need(*a) && rEOF);   ++a;  
        while ((*a = nc()) && need(*a) && rEOF)++a; --p1, *a = '\0';  
    }  
}using namespace _;


const int N = 5e6 + 10;

int n;
int p[N];

int find(int x) {
    if (p[x] == x) return x;
    return p[x] = find(p[x]);
}

int main () {
    int n;
    read(n);
    
    for (int i = 1; i <= n; ++ i) p[i] = i;
    
    while (n --) {
        int opt, x;
        read(opt); read(x);
        if (opt == 1) {
            p[find(x)] = find(x + 1);
        } else {
            int i = find(1);
            if (i == x) i = find(x + 1);
            printf ("%d\n", i);
        }
    }
    return 0;
}
```



### 格子游戏

> 通过将二维坐标一维化使得当形成一个环的时候即游戏结束

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 40000 + 10;

int n, m;
int p[N];

int find(int x) {
    if (x == p[x]) return x;
    return p[x] = find(p[x]);
}

int get(int x, int y) {
    return x * n + y;
}

void solve() {
    scanf("%d%d", &n, &m);
    
    for (int i = 0; i < n * n; ++ i) p[i] = i;
    
    for (int i = 1; i <= m; ++ i) {
        int x, y; char dir;
        scanf("%d%d %c", &x, &y, &dir);
        x --; y --;
        
        int a = get(x, y);
        int b;
        
        if (dir == 'D') {
            b = get(x + 1, y);
        } else {
            b = get(x, y + 1);
        }
        
        int pa = find(a), pb = find(b);
        
        if (pa == pb) {
            printf("%d\n", i);
            return ;
        } else {
            p[pa] = pb;
        }
    }
    printf("draw\n");
}

int main() {
    solve();
    return 0;
}
```



### 搭配购买

> 可以看成配套商品必须是一个连通块，然后将整个连通块看成一个新的商品，用01背包跑一遍即可得到答案

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 10000 + 10;

int n, m, w;
int c[N], d[N];
int p[N];
int f[N];

int find(int x) {
    if (x == p[x]) return x;
    return p[x] = find(p[x]);
}

void solve() {
    scanf("%d%d%d", &n, &m, &w);
    
    for (int i = 1; i <= n; ++ i) p[i] = i;
    
    for (int i = 1; i <= n; ++ i) {
        scanf("%d%d", &c[i], &d[i]);
    }
    for (int i = 1; i <= m; ++ i) {
        int a, b;
        scanf("%d%d", &a, &b);
        int pa = find(a), pb = find(b);
        if (pa != pb) {
            p[pa] = pb;
            c[pb] += c[pa];
            d[pb] += d[pa];
        }
    }
    
    for (int i = 1; i <= n; ++ i) {
        int pa = find(i);
        if (pa == i) {
            for (int j = w; j >= c[i]; -- j) {
                f[j] = max(f[j], f[j - c[i]] + d[i]);
            }
        }
    }
    
    printf("%d\n", f[w]);
}

int main() {
    solve();
    return 0;
}
```



### 程序自动分析

> 容易知道关系之间的顺序是没够联系的，所以可以先把等价的合并在一起，然后再判不等价的是否矛盾即可
>
> 数据范围较大，需要离散化

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;

int m;
int idx;
int p[N << 1];
unordered_map<int, int> mp;
struct query{
    int x, y, o;
} querys[N];

int find(int x) {
    if (x == p[x]) return x;
    return p[x] = find(p[x]);
}

int get(int x) {
    if (!mp.count(x)) 
        mp[x] = ++ idx;

    return mp[x];
}

void solve() {
    scanf("%d", &m);
    
    idx = 0;
    mp.clear();
    
    for (int i = 1; i <= m; ++ i) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        querys[i] = {get(a), get(b), c};
    }
    
    for (int i = 1; i <= idx; ++ i) p[i] = i;
    
    for (int i = 1; i <= m; ++ i) {
        int a = querys[i].x, b = querys[i].y, op = querys[i].o;
        if(op == 1) {
            int pa = find(a), pb = find(b);
            p[pa] = pb;
        }
    }
    
    for (int i = 1; i <= m; ++ i) {
        int a = querys[i].x, b = querys[i].y, op = querys[i].o;
        if(op == 0) {
            int pa = find(a), pb = find(b);
            if (pa == pb) {
                puts("NO");
                return ;
            }
        }
    }
    puts("YES");
}

int main() {
    int T;
    scanf("%d", &T);
    while (T --) 
        solve();
    return 0;
}
```



### 奇偶游戏

> 01序列统计01个数考虑前缀和，$S_i$ 表示从$[1, i]$ 区间中1的个数
>
> 偶数个1：$S_R 与 S_{L - 1}$ 奇偶性质相同，否则不同
>
> 此题带权并查集类似食物链

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 4e4 + 10;

int n, m;
int p[N], d[N];
char type[110];
unordered_map<int, int> S;

int get(int x) {
    if (!S.count(x)) S[x] = ++ n;
    return S[x];
}

int find(int x) {
    if (p[x] != x) {
        int root = find(p[x]);
        d[x] ^= d[p[x]];
        p[x] = root;
    }
    return p[x];
}

void solve() {
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i < N; ++ i) p[i] = i;
    
    n = 0;
    for (int i = 1; i <= m; ++ i) {
        int a, b;
        scanf("%d%d %s", &a, &b, type);
        
        int t = type[0] == 'o';
        
        a = get(a - 1); b = get(b);
        
        int pa = find(a), pb = find(b);
        
        if (pa == pb) {
            if ((d[a] ^ d[b]) != t) {
                printf("%d", i - 1);
                return ;
            }
        } else {
            p[pa] = pb;
            d[pa] = d[a] ^ d[b] ^ t;
        }
    }
    printf("%d\n", m);
}

int main() {
    solve();
    return 0;
}
```



### 食物链

> 假设有三只动物x, y, z，x吃y，y吃z，则可以推出x吃z，可以通过带权并查集维护之间的关系，0为同类，1为强，2为弱
>
> 路径维护的时候要注意顺序! p[pb] = pa! 就得更新pb

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 50000 + 10;

int n, m;
int p[N], d[N];

int find(int x) {
    if (x != p[x]) {
        int oldParent = p[x];
        p[x] = find(p[x]);
        d[x] = (d[x] + d[oldParent]) % 3;
    }
    return p[x];
}

bool check(int x, int y) {
    return (x % 3) == (y % 3);
}

int cmod(int x) {
    return (x + 3) % 3;
}

void solve() {
    scanf("%d%d", &n, &m);
    for (int i = 0; i <= n; ++ i) p[i] = i;
    
    int ans = 0;
    for (int i = 1; i <= m; ++ i) {
        int t, x, y;
        scanf("%d%d%d", &t, &x, &y);
        if (x > n || y > n) ans ++;
        else {
            int pa = find(x), pb = find(y);
            if (pa == pb) {
                ans += 1 - check(d[x], d[y] + (t == 2));
            } else {
                p[pb] = pa;
                d[pb] = cmod(d[x] - d[y] - (t == 2));
            }
        }
    }
    
    printf("%d", ans);
}

int main() {
    solve();
    
    return 0;
}
```

