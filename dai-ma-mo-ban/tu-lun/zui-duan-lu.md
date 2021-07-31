# 最短路

## Floyd全源最短路

```cpp
for (int k = 1; k <= n; ++ i)
    for (int i = 1; i <= n; ++ i)
        for (int j = 1; j <= n; ++ j)
                f[i][j] = min(f[i][j], f[i][k] + f[k][j]);
```

## Dijkstra单源最短路

> 跑得快，但是不支持负权图

```cpp
const int N = 100000 + 10; //点数
const int M = 100000 + 10; //边数
const int INF = 0x3f3f3f3f; //最大值

struct qnode {
    int u , v;
    qnode (int _u , int _c): u(_u) , c(_c) {}
    bool operator < (const qnode &r) const {
        return c > r.c;
    }
}
int dist[N] , vis[N];
void dijkstra (int s , int n) {
    priority_queue <qnode> que;
    for (int i = 1 ; i <= n ; ++ i) 
        dist[i] = INF, vis[i] = 0;
    que.push (qnode (s , 0));
    dist[s] = 0;
    while (!que.empty ()) {
        int u = que.top().u;
        que.pop ();
        if (vis[u]) continue;
        vis[u] = 1;
        for (int i = head[u] ; ~i ; i = next[i]) {
            int v = to[i] , w = cost[i];
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                if (!vis[v]) q.push (qnode (v , dist[v]));
            }
        }
    }
}
```

## SPFA

> 好写，但是容易被卡

```cpp
const int N = 100000 + 10; //点数
const int M = 100000 + 10; //边数
const int INF = 0x3f3f3f3f; //最大值

int dist[N] , cnt[N] , vis[N];
bool SPFA (int s , int n) {
    queue <int> que;
    for (int i = 1 ; i <= n ; ++ i) 
        dist[i] = INF , cnt[i] = vis[i] = 0;
    dist[s] = 0; vis[s] = 1; cnt[s] = 1;
    que.push (s);
    while (!que.empty ()) {
        int u = que.front ();
        que.pop ();
        vis[u] = false;
        for (int i = head[u] ; ~i ; i = next[i]) {
            int v = to[v] , w = cost[v];
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                if (!vis[v]) {
                    vis[v] = 1;
                    que.push (v);
                    if (++cnt[v] > n) return false;
                }
            }
        }
    }
    return true;
}
```

## Johnson全源最短路

> 通过SPFA枚举虚拟结点顺带判负环，然后使边权为正运行dijkstra求最短路

```cpp
const int N = 100000 + 10;
const int M = 100000 + 10 + N;
const int INF = 0x3f3f3f3f;

struct qnode {
    int u , v;
    qnode (int _u , int _c): u(_u) , c(_c) {}
    bool operator < (const qnode &r) const {
        return c > r.c;
    }
}
int dist[N] , vis[N] , cnt[N] , h[N];
bool SPFA (int s , int n) {
    queue <int> que;
    for (int i = 1 ; i <= n ; ++ i) 
        h[i] = INF , cnt[i] = vis[i] = 0;
    dist[s] = 0; vis[s] = 1; cnt[s] = 1;
    que.push (s);
    while (!que.empty ()) {
        int u = que.front ();
        que.pop ();
        for (int i = head[u] ; ~i ; i = next[i]) {
            int v = to[v] , w = cost[v];
            if (h[v] > h[u] + w) {
                h[v] = h[u] + w;
                if (!vis[v]) {
                    vis[v] = 1;
                    que.push (v);
                    if (++cnt[v] > n) return false;
                }
            }
        }
    }
    return true;
}
void dijkstra (int s , int n) {
    priority_queue <qnode> que;
    for (int i = 1 ; i <= n ; ++ i) 
        dist[i] = INF, vis[i] = 0;
    que.push (qnode (s , 0));
    dist[s] = 0;
    while (!que.empty ()) {
        int u = que.top().u;
        que.pop ();
        if (vis[u]) continue;
        vis[u] = 1;
        for (int i = head[u] ; ~i ; i = next[i]) {
            int v = to[i] , w = cost[i];
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                if (!vis[v]) q.push (qnode (v , dist[v]));
            }
        }
    }
}
bool johnson (int n) {
    for (int i = 1 ; i <= n ; ++ i) add (0 , i , 0);
    if (!SPFA (0,n)) return false;
    for (int i = 1 ; i <= n ; ++ i) 
        for (int j = head[i] ; ~j ; j = next[j]) cost[j] += h[i] - h[to[j]];
    for (int i = 1 ; i <= n ; ++ i) {
        dijkstra (i , n); //分别找某个点去终点的最短路
       /* 如果dist[p] == INF , 说明到不了 */
       /* 否则最短路径为dist[p] + h[p] - h[i]; */
    }
    return true;
}
```

