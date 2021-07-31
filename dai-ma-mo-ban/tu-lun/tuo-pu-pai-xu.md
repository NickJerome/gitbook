# 拓扑排序

> 通过对图的入读是否为0进行对点排序

```cpp
#include <iostream>
#include <cstring>
#include <queue>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10, M = 1e5 + 10;

int n, m;
int h[N], e[M], ne[M], idx;
int in[N], que[N];

void add (int u, int v) {
    e[idx] = v;
    ne[idx] = h[u];
    h[u] = idx ++;
}

bool topsort () {
    int hh = 0, tt = -1;

    for (int i = 1; i <= n; ++ i) 
        if (!in[i]) que[++ tt] = i;

    while (hh <= tt) {
        int t = que[hh ++];

        for (int i = h[t]; ~i; i = ne[i]) {
            int v = e[i];
            if (-- in[v] == 0) que[++ tt] = v;
        }
    }

    return tt == n - 1;
}

int main () {
    scanf ("%d%d", &n, &m);
    memset (h, -1, sizeof h);

    for (int i = 1; i <= m; ++ i) {
        int a, b;
        scanf ("%d%d", &a, &b);
        add (a, b);
        in[b]++;
    }
    if (!topsort ()) puts("-1");
    else {
        for (int i = 0; i < n; ++ i) 
            printf ("%d ", que[i]);
    }
    return 0;
}
```

