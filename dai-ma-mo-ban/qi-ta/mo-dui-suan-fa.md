# 莫队算法

## 普通莫队

> 离线查找区间计数问题

```cpp
#include <cmath>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 5e4 + 10;

int n, m, k;
int w[N], len;

struct Query{
    int id, l, r;
} q[N];
int cnt[N];
long long ans[N];

bool cmp(const Query &a, const Query &b) {
    int i = a.l / len, j = b.l / len;
    if (i != j) return i < j;
    return a.r < b.r;
}

void add(int x, long long &res) {
    res -= cnt[x] * cnt[x];
    cnt[x] ++;
    res += cnt[x] * cnt[x];
}
void del(int x, long long &res) {
    res -= cnt[x] * cnt[x];
    cnt[x] --;
    res += cnt[x] * cnt[x];
} 

int main () {
    scanf("%d%d%d", &n, &m, &k);
    len = ceil(sqrt(n * 1.0));
    for (int i = 1; i <= n; ++ i) scanf("%d", &w[i]);
    for (int i = 1; i <= m; ++ i) {
        scanf ("%d%d", &q[i].l, &q[i].r); q[i].id = i;
    }
    sort(q + 1, q + 1 + m, cmp);

    long long res = 0;
    for (int k = 1, i = 0, j = 1; k <= m; ++ k) {
        int id = q[k].id, l = q[k].l, r = q[k].r;
        while (i < r) add(w[++ i], res);
        while (i > r) del(w[i --], res);
        while (j < l) del(w[j ++], res);
        while (j > l) add(w[-- j], res);
        ans[id] = res;
    }

    for (int i = 1; i <= m; ++ i) printf ("%d\n", ans[i]);

    return 0;
}
```

## 带修改莫队

> 添加一维时间维度，用于添加修改或者还原修改

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


const int N = 133333 + 10;

struct Query {
    int id, l, r, t;
} q[N];

struct Modify {
    int id, p, v;
} c[N];

int n, m;
int w[N], cnt[N];
int mq, mc;
int ans[N];
int len;

void add(int x, int &res) {
    if (!cnt[x]) res ++;
    cnt[x] ++;
}
void del(int x, int &res) {
    cnt[x] --;
    if (!cnt[x]) res --;
}

bool cmp (const Query &a, const Query &b) {
    int al = a.l / len, ar = a.r / len;
    int bl = b.l / len, br = b.r / len;
    if (al != bl) return al < bl;
    if (ar != br) return ar < br;
    return a.t < b.t;
}

int main () {
    read(n); read(m);
    for (int i = 1; i <= n; ++ i) read(w[i]);
    len = pow(n,0.75)+1;

    for (int i = 1; i <= m; ++ i) {
        char opt;
        scanf(" %c", &opt);
        if (opt == 'Q') mq ++, read(q[mq].l), read(q[mq].r), q[mq].t = mc, q[mq].id = mq;
        else mc ++, read(c[mc].p), read(c[mc].v);
    }

    sort(q + 1, q + 1 + mq, cmp);

    int res = 0;
    for (int k = 1, i = 0, j = 1, t = 0; k <= mq; ++ k) {
        int id = q[k].id, l = q[k].l, r = q[k].r, mt = q[k].t;
        while (i < r) add(w[++ i], res);
        while (i > r) del(w[i --], res);
        while (j < l) del(w[j ++], res);
        while (j > l) add(w[-- j], res);

        while (t < mt) {
            t ++;
            if (j <= c[t].p && c[t].p <= i) {
                del(w[c[t].p], res);
                add(c[t].v, res);
            }
            swap(w[c[t].p], c[t].v);
        }
        while (t > mt) {
            if (j <= c[t].p && c[t].p <= i) {
                del(w[c[t].p], res);
                add(c[t].v, res);
            }
            swap(w[c[t].p], c[t].v);
            t --;
        }

        ans[id] = res;
    }

    for(int i = 1; i <= mq; ++ i) printf ("%d\n", ans[i]);
    return 0;
}
```

## 树上莫队

