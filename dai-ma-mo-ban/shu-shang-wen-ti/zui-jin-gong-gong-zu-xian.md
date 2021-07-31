# 最近公共祖先

## 倍增法

> 通过两倍两倍的跳来减少查询次数

### BFS

```cpp
#include <set>
#include <map>
#include <cmath>
#include <deque>
#include <queue>
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> PII;

template <typename T>
inline void read(T &x) {
    x = 0; T w = 1; char ch = getchar ();
    while (ch < '0' || ch > '9') { if (ch == '-') w = -1; ch = getchar(); }
    while (ch >= '0' && ch <= '9') { x = (x << 3) + (x << 1) + (ch & 15); ch = getchar(); }
    x *= w;
}
template <typename T>
inline void print(T x) {
    if (x < 0) putchar ('-'), x = -x;
    if (x < 10) putchar (x + 48);
    else print (x / 10), putchar ((x % 10) + 48);
}

const int N = 4e4 + 10, M = 2 * N;
const int LogN = 30;

int n, m;
int h[N], e[M], ne[M], idx;
int dep[N], fa[N][LogN];
int q[N];

void add(int a, int b) {
    e[idx] = b; ne[idx] = h[a]; h[a] = idx ++;
}

void bfs(int root) {
    memset(dep, 0x3f, sizeof dep);
    dep[0] = 0; dep[root] = 1;
    int hh = 0, tt = -1;
    q[++ tt] = root;
    while (hh <= tt) {
        int u = q[hh ++];
        for (int i = h[u]; ~i; i = ne[i]) {
            int v = e[i];
            if (dep[v] > dep[u] + 1) {
                dep[v] = dep[u] + 1;
                q[++ tt] = v;
                fa[v][0] = u;
                for (int k = 1; k < LogN; ++ k) 
                    fa[v][k] = fa[fa[v][k - 1]][k - 1];
            }
        }
    }
}

int lca (int a, int b) {
    if (dep[a] < dep[b]) swap(a, b);
    for (int k = LogN - 1; k >= 0; -- k) 
        if (dep[fa[a][k]] >= dep[b]) a = fa[a][k];
    if (a == b) return a;
    for (int k = LogN - 1; k >= 0; -- k) 
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main () {
    int root = 0;
    memset (h, -1, sizeof h);

    read(n);
    for (int i = 1; i <= n; ++ i) {
        int a, b;
        read(a); read(b);
        if (b == -1) root = a;
        else {
            add (a, b); add (b, a);
        }
    }

    bfs (root);

    scanf ("%d", &m);
    while (m --) {
        int a, b;
        read(a); read(b);
        int p = lca(a, b);
        if (p == a) puts ("1");
        else if (p == b) puts("2");
        else puts("0");
    }
    return 0;
}
```

### DFS

```cpp
#include <set>
#include <map>
#include <cmath>
#include <deque>
#include <queue>
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> PII;

template <typename T>
inline void read(T &x) {
    x = 0; T w = 1; char ch = getchar ();
    while (ch < '0' || ch > '9') { if (ch == '-') w = -1; ch = getchar(); }
    while (ch >= '0' && ch <= '9') { x = (x << 3) + (x << 1) + (ch & 15); ch = getchar(); }
    x *= w;
}
template <typename T>
inline void print(T x) {
    if (x < 0) putchar ('-'), x = -x;
    if (x < 10) putchar (x + 48);
    else print (x / 10), putchar ((x % 10) + 48);
}

const int N = 4e4 + 10, M = 2 * N;
const int LogN = 30;

int n, m;
int h[N], e[M], ne[M], idx;
int dep[N], fa[N][LogN];
int q[N];

void add(int a, int b) {
    e[idx] = b; ne[idx] = h[a]; h[a] = idx ++;
}

void dfs (int u, int v) {
    dep[v] = dep[u] + 1;
    fa[v][0] = u;
    for (int k = 1; k < LogN; ++ k) 
        fa[v][k] = fa[fa[v][k - 1]][k - 1];

    for (int i = h[v]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dep[j])
            dfs (v, j);
    }
}

int lca (int a, int b) {
    if (dep[a] < dep[b]) swap(a, b);
    for (int k = LogN - 1; k >= 0; -- k) 
        if (dep[fa[a][k]] >= dep[b]) a = fa[a][k];
    if (a == b) return a;
    for (int k = LogN - 1; k >= 0; -- k) 
        if (fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}

int main () {
    int root = 0;
    memset (h, -1, sizeof h);

    read(n);
    for (int i = 1; i <= n; ++ i) {
        int a, b;
        read(a); read(b);
        if (b == -1) root = a;
        else {
            add (a, b); add (b, a);
        }
    }

    dfs (0, root);

    scanf ("%d", &m);
    while (m --) {
        int a, b;
        read(a); read(b);
        int p = lca(a, b);
        if (p == a) puts ("1");
        else if (p == b) puts("2");
        else puts("0");
    }
    return 0;
}
```

## Tarjan

> 通过并查集记录某一个点的祖先结点\[离线算法\]

```cpp
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;
typedef pair<int, int> PII;

const int N = 4e4 + 10;
const int M = 2 * N;

int n, m;
int h[N], e[M], ne[M], idx;
int pa[N];
int vis[N];
vector<PII> query[N];
PII s[N];
int lca[N];

void add(int a, int b) {
    e[idx] = b; ne[idx] = h[a]; h[a] = idx ++;
}
void init () {
    for (int i = 1; i < N; ++ i)
        pa[i] = i;
    memset(h, -1, sizeof h);
}
int find(int x) {
    if (pa[x] == x) return x;
    else return pa[x] = find(pa[x]);
}
void tarjan(int u) {
    vis[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int v = e[i];
        if (vis[v]) continue;
        tarjan (v);
        pa[v] = u;
    }
    for (auto item : query[u]) {
        int v = item.first, id = item.second;
        if (vis[v] == 2) {
            lca[id] = find(v);
        }
    }
    vis[u] = 2;
}

int main () {
    scanf("%d", &n);

    int root = 0;
    init ();

    for (int i = 1; i <= n; ++ i) {
        int a, b;
        scanf ("%d%d", &a, &b);
        if (b == -1) root = a;
        else add(a, b), add(b, a);
    }

    scanf("%d", &m);
    for (int i = 1; i <= m; ++ i) {
        int a, b;
        scanf("%d%d", &a, &b);
        query[a].push_back({b, i});
        query[b].push_back({a, i});
        s[i] = {a, b};
    }

    tarjan(root);

    for (int i = 1; i <= m; ++ i) {
        int a = s[i].first, b = s[i].second;
        if (lca[i] == a) puts("1");
        else if (lca[i] == b) puts("2");
        else puts("0");
    }

    return 0;
}
```

