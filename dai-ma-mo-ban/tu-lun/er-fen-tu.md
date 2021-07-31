# 二分图

## 染色法判定二分图

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e5 + 10;
const int M = 2e5 + 10;

int n, m;
int h[N], e[M], ne[M], idx;
int color[N];
void add(int a, int b) {
    e[idx] = b; ne[idx] = h[a]; h[a] = idx ++;
}

bool dfs(int u, int c) {
    color[u] = c;
    for (int i = h[u]; ~i; i = ne[i]) {
        int v = e[i];
        if (!color[v]) {
            if (!dfs (v, 3 - c)) return false;
        }
        else if (color[v] == c) return false;
    }
    return true;
}

int main () {
    memset (h, -1, sizeof h);
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= m; ++ i) {
        int u, v;
        scanf("%d%d", &u, &v);
        add(u, v); add(v, u);
    }
    bool flag = true;
    for (int i = 1; i <= n; ++ i) 
        if (!color[i])
            if (!dfs (i, 1)) {
                flag = false;
                break;
            }
    if (flag) puts ("Yes");
    else puts ("No");
    return 0;
}
```

## 二分图的最大匹配

### 匈牙利算法

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 500 + 10, M = 1e5 + 10;

int n1, n2, m;
int h[N], e[M], ne[M], idx;
int match[N];
bool vis[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

bool find(int u) {
    for (int i = h[u]; ~i; i = ne[i]) {
        int v = e[i];
        if (!vis[v]) {
            vis[v] = true;
            if (!match[v] || find(match[v])) {
                match[v] = u;
                return true;
            }
        }
    }
    return false;
}

int main() {
    scanf("%d%d%d", &n1, &n2, &m);
    memset(h, -1, sizeof h);
    for (int i = 1; i <= m; ++ i) {
        int u, v;
        scanf("%d%d", &u, &v);
        add(u, v);
    }
    int res = 0;
    for (int i = 1; i <= n1; ++ i) {
        memset(vis, false, sizeof vis);
        if (find(i)) res ++;
    }
    printf ("%d\n", res);
    return 0;
}
```

