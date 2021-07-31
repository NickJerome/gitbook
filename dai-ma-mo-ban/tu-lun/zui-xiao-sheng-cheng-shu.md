# 最小生成树

## Prim 算法

> 通过贪心思想，每次求能连接集合的最小边，然后加入集合当中
>
> 朴素版时间复杂度：O\(n^2\) 【用于稠密图】
>
> 堆优化版时间复杂度：O\(nlogm\)【用于稀疏图】【不常用，推荐Kruskal】

### 朴素版

```cpp
//求最小生成树的总权值
int prim () {
    memset (dist, 0x3f, sizeof (dist));
    int res = 0;
    dist[1] = 0;
    for (int i = 1; i <= n; ++ i) {
        int t = -1;
        for (int j = 1; j <= n; ++ j)
            if (!vis[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
        if (dist[t] == INF) return INF;
        res += dist[t];
        vis[t] = true;

        for (int j = 1; j <= n; ++ j) dist[j] = min(dist[j], g[t][j]);
    }

    return res;
}
```

### 堆优化版

```cpp
struct qnode{
    int p, w;
    qnode (int _p, int _w): p(_p), w(_w) {}
    bool operator < (const qnode &r) const {
        return w > r.w;
    }
}

int prim () {
    memset (dist, 0x3f, sizeof (dist));

    int res = 0;
    dist[1] = 0;

    priority_queue<qnode> que;
    que.push (qnode (1, 0));

    for (int i = 1; i <= n; ++ i) {
        while (!que.empty() && vis[que.top().p]) que.pop();

        if (que.empty()) return INF;
        int t = que.top().p;
        que.pop ();

        res += dist[t];

        for (int j = 1; j <= n; ++ j) {
            if (!vis[j] && dist[j] > g[t][j]) {
                dist[j] = g[t][j];
                que.push (qnode(j, dist[j]));
            }
        }
    }

    return res;
}
```

### 手写二叉堆

```cpp
int n, m;
int g[N][N];
int dist[N], vis[N];
int h[N], hp[N], ph[N], s; //ph是点i在堆里的位置，hp是位置i是哪个点

void heap_swap (int a, int b) {
    swap (ph[hp[a]], ph[hp[b]]);
    swap (hp[a], hp[b]);
    swap (h[a], h[b]);
}

void up (int u) {
    while (u / 2 && h[u / 2] > h[u]) {
        heap_swap (u, u / 2);
        u /= 2;
    }
}

void down (int u) {
    int t = u;
    if (u * 2 <= s && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= s && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (t != u) {
        heap_swap (u, t);
        down (t);
    }
}

int prim () {
    memset (dist, 0x3f, sizeof (dist));

    int res = 0;
    dist[1] = 0;
    s = n;
    for (int i = 1; i <= n; ++ i) {
        ph[i] = hp[i] = i; h[i] = dist[i];
    }

    for (int i = n / 2; i; -- i) down (i);

    for (int i = 1; i <= n; ++ i) {
        int t = hp[1];
        heap_swap(1, s --); down(1);

        if (dist[t] == INF) return INF;

        res += dist[t];
        vis[t] = true;

        for (int j = 1; j <= n; ++ j) 
            if (!vis[j] && dist[j] > g[t][j]) {
                h[ph[j]] = dist[j] = g[t][j];
                up (ph[j]);
            }
    }

    return res;
}
```

## Kruskal算法

> 通过当前能得到的最小边权，然后判断这条边的两个端点是否在一个连通块里，如果在就不更新，否则加入联通块。

```cpp
struct Edge {
    int u, v, w;
    bool operator < (const Edge &r) const {
        return w > r.w;
    }
}edge[M];

int find (int x) {
    if (fa[x] != x) fa[x] = find(fa[x]);
    return fa[x];
}

int kruskal () {
    sort (edge, edge + m);

    for (int i = 1; i <= n; ++ i) fa[i] = i;

    int res = 0, cnt = 0;

    for (int i = 1; i <= m; ++ i) {
        int u = edge[i].u, v = edge[i].v, w = edge[i].w;
        u = find(u); v = find(v);
        if (u != v) {
            fa[u] = v;
            res += w;
            cnt ++;
        }
    }
    if (cnt < n - 1) return INF;
    return res;
}
```

