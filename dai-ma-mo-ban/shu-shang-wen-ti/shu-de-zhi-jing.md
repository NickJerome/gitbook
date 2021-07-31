# 树的直径

## 两次DFS

> 随便一个点，找到离他最远的点，再从最远的点出发，找到离这个点最远的点，这两个点的距离就是树的直径 证明：无 Tips：不支持负权图

```cpp
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e4 + 10;

int n, c, d[N];
vector<int> G[N];

void dfs(int u, int fa) {
    for (int v : G[u]) {
        if (v == fa) continue;
        d[v] = d[u] + 1;
        if(d[v] > d[c]) c = v;
        dfs(v, u);
    }
}

int main () {
    scanf("%d", &n);
    for (int i = 1; i < n; ++ i) {
        int u, v;
        scanf("%d%d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    dfs(1, 0);
    d[c] = 0;
    dfs(c, 0);
    printf("%d\n", d[c]);
    return 0;
}
```

## 两次BFS

> 同DFS差不多

```cpp
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e4 + 10;
const int INF = 1e9;

int n, c, d[N];
vector<int> G[N];
int q[N];

void bfs(int u) {
    for(int i = 1; i <= n; ++ i) d[i] = INF;
    int hh = 0, tt = -1;
    q[++ tt] = u; 
    d[u] = 0;
    c = u;

    while (hh <= tt) {
        int u = q[hh ++];
        for (int v : G[u]) {
            if (d[v] == INF) {
                d[v] = d[u] + 1;
                if (d[c] < d[v]) c = v;
                q[++ tt] = v;
            }
        }
    }
}

int main () {
    scanf("%d", &n);
    for (int i = 1; i < n; ++ i) {
        int u, v;
        scanf("%d%d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    bfs(1);
    bfs(c);
    printf("%d\n", d[c]);
    return 0;
}
```

## 树上DP

> 我们记录当 $1$ 为树的根时，每个节点作为子树的根向下，所能延伸的最远距离 $d\_1$，和次远距离 $d\_2$，那么直径就是所有 $d\_1 + d\_2$ 的最大值。

```cpp
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e4 + 10;
const int INF = 1e9;

int n, d = 0;
int d1[N], d2[N];
vector<int> G[N];

void dfs(int u, int fa) {
      d1[u] = d2[u] = 0;

      for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        int t = d1[v] + 1;
        if (t > d1[u])
              d2[u] = d1[u], d1[u] = t;
        else if (t > d2[u])
              d2[u] = t;
      }
      d = max(d, d1[u] + d2[u]);
}

int main() {
      scanf("%d", &n);
      for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
      }
      dfs(1, 0);
      printf("%d\n", d);
      return 0;
}
```

