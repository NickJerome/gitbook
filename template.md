# 字符串



## KMP

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;
const int M = 1e6 + 10;

int n, m;
char p[N], s[M];
int ne[N];

void solve() {
    scanf("%d%s", &n, p + 1);
    scanf("%d%s", &m, s + 1);
    
    //kmp 预处理
    for(int i = 2, j = 0; i <= n; ++ i) {
        while (j && p[i] != p[j + 1]) j = ne[j];
        if (p[i] == p[j + 1]) ++ j;
        ne[i] = j;
    }
    
    //kmp匹配
    for(int i = 1, j = 0; i <= m; ++ i) {
        while (j && s[i] != p[j + 1]) j = ne[j];
        if (s[i] == p[j + 1]) ++ j;
        
        if (j == n) {
            printf("%d ", i - n);
            j = ne[j];
        }
    }
}

int main() {
    solve();
    
    return 0;
}
```



## Trie树

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 2e4 + 10;
const int M = 1e5 + 10;

int n;
int tr[N][26], tot;
int cnt[N];
char s[M];

int New() {
    int p = ++ tot;
    memset(tr[p], 0, sizeof (tr[p]));
    return p;
}

void add() {
    int p = 0;
    for (int i = 1; s[i]; ++ i) {
        int u = s[i] - 'a';
        if(!tr[p][u]) tr[p][u] = New();
        p = tr[p][u];
    }
    cnt[p] ++;
}

int ask() {
    int p = 0;
    for (int i = 1; s[i]; ++ i) {
        int u = s[i] - 'a';
        if (!tr[p][u]) return 0;
        p = tr[p][u];
    }
    return cnt[p];
}

void solve() {
    scanf("%d", &n);
    while (n --) {
        char op;
        scanf(" %c%s", &op, s + 1);
        if (op == 'I') add();
        else printf("%d\n", ask());
    }
}

int main() {
    solve();
    
    return 0;
}
```



## 字符串Hash

> 建议采用多哈希减少哈希碰撞概率

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long uLL;

const int N = 1e5 + 10;
const int P[3] = {131, 13331, 1333331};

int n, m;
uLL p[N][3], h[N][3];
char s[N];

uLL get(int l, int r, int t) {
    return h[r][t] - h[l - 1][t] * p[r - l + 1][t];
}

void solve() {
    scanf("%d%d", &n, &m);
    scanf("%s", s + 1);
    
    for (int t = 0; t < 3; ++ t) {
        p[0][t] = 1;
        for (int i = 1; i <= n; ++ i) {
            h[i][t] = h[i - 1][t] * P[t] + s[i];
            p[i][t] = p[i - 1][t] * P[t];
        }
    }
    
    while (m -- ) {
        int l1, l2, r1, r2;
        scanf("%d%d%d%d", &l1, &r1, &l2, &r2);
        
        bool flag = true;
        for (int i = 0; i < 3; ++ i) {
            if (get(l1, r1, i) != get(l2, r2, i)) flag = false;
        }
        
        if (flag) {
            puts("Yes");
        } else {
            puts("No");
        }
    }
}

int main() {
    solve();
    
    return 0;
}
```





# 数据结构



## 01树

> 可以用来解决关于异或的问题

> 例题：最大异或和

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 3e6 + 10;

int n;
int tr[N][2], tot;
int a[N];

int New() {
    int p = ++ tot;
    memset(tr[p], 0, sizeof (tr[p]));
    return p;
}

void add(int x) {
    int p = 0;
    for (int i = 30; i >= 0; -- i) {
        int u = (x >> i) & 1;
        if (!tr[p][u]) tr[p][u] = New();
        p = tr[p][u];
    }
}

int ask(int x) {
    int p = 0;
    int res = 0;
    for (int i = 30; i >= 0; -- i) {
        int u = (x >> i) & 1;
        if (tr[p][!u]) {
            res += (1 << i);
            u = 1 - u;
        }
        p = tr[p][u];
    }
    return res;
}

void solve() {
    scanf("%d", &n);
    for(int i = 1; i <= n; ++ i) {
        scanf("%d", &a[i]);
        add(a[i]);
    }
    
    int res = 0;
    for (int i = 1; i <= n; ++ i) {
        res = max(res, ask(a[i]));
    }
    
    printf("%d", res);
}

int main() {
    solve();
    
    return 0;
}
```



## 主席树



### 静态区间第k小

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;

struct Node {
    int l, r;
    int s;
} tr[N * 20];
int root[N], tot;
int w[N], d[N];

int build(int l, int r) {
    int p = ++ tot;
    int m = l + r >> 1;
    if (l < r) {
        tr[p].l = build(l, m);
        tr[p].r = build(m + 1, r);
    }
    tr[p].s = 0;
    return p;
}

int add(int p, int l, int r, int x, int c) {
    int q = ++ tot;
    tr[q] = tr[p];
    int m = l + r >> 1;
    tr[q].s = tr[p].s + c;
    if (l < r) {
        if (x <= m) {
            tr[q].l = add(tr[p].l, l, m, x, c);
        } else {
            tr[q].r = add(tr[p].r, m + 1, r, x, c);
        }
    }
    return q;
}

int query(int p, int q, int l, int r, int x) {
    if (l == r) return l;
    
    int cnt = tr[tr[q].l].s - tr[tr[p].l].s;
    int mid = (l + r) / 2;
    
    if (x <= cnt) return query(tr[p].l, tr[q].l, l, mid, x);
    else return query(tr[p].r, tr[q].r, mid + 1, r, x - cnt);
}

int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) {
        scanf("%d", &w[i]);
        d[i] = w[i];
    }
    
    sort(d + 1, d + 1 + n);
    int len = unique(d + 1, d + 1 + n) - (d + 1);
    for(int i = 1; i <= n; i++) 
        w[i] = lower_bound(d + 1, d + 1 + len, w[i]) - d;
    
    root[0] = build(1, len);
    for (int i = 1; i <= n; ++ i) {
        root[i] = add(root[i - 1], 1, len, w[i], 1);
    }
    
    while (m --) {
        int l, r, k;
        scanf("%d%d%d", &l, &r, &k);
        printf("%d\n", d[query(root[l - 1], root[r], 1, len, k)]);
    }
    
    return 0;
}
```



### 静态区间不同数统计

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e6 + 10;

struct Node {
    int l, r;
    int s;
} tr[N * 40];
int root[N], tot;

int build(int l, int r) {
    int p = ++ tot;
    int m = (l + r) >> 1;
    if (l < r) {
        tr[p].l = build(l, m);
        tr[p].r = build(m + 1, r);
    }
    tr[p].s = 0;
    return p;
}

int add(int p, int l, int r, int x, int c) {
    int q = ++ tot;
    tr[q] = tr[p];
    tr[q].s += c;
    int m = (l + r) >> 1;
    if (l < r) {
        if (x <= m) {
            tr[q].l = add(tr[p].l, l, m, x, c);
        } else {
            tr[q].r = add(tr[p].r, m + 1, r, x, c);
        }
    }
    return q;
}

int query(int p, int l, int r, int x) {
    if (l == r) return tr[p].s;
    
    int m = (l + r) >> 1;
    if (x <= m) return tr[tr[p].r].s + query(tr[p].l, l, m, x);
    else return query(tr[p].r, m + 1, r, x);
}

int n, m;
int a[N];
int pos[N];

void solve() {
    cin >> n;
    for (int i = 1; i <= n; ++ i) {
        cin >> a[i];
    }
    
    root[0] = build(1, n);
    
    for (int i = 1; i <= n; ++ i) {
        if (pos[a[i]] == 0) {
            root[i] = add(root[i - 1], 1, n, i, 1);
        } else {
            int tmp = add(root[i - 1], 1, n, pos[a[i]], -1);
            root[i] = add(tmp, 1, n, i, 1);
        }
        pos[a[i]] = i;
    }
    
    cin >> m;
    while (m -- ) {
        int l, r;
        cin >> l >> r;
        cout << query(root[r], 1, n, l) << "\n";
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    
    solve();
    
    return 0;
}
```



### 查询区间[l, r]满足值在[ql,qr]范围内的个数

> 例题：Eyjafjalla

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;
const int M = 2e5 + 10;
const int C = 1e9 + 1;

int n, q;
int h[N], e[M], ne[M], idx;
int w[N], d[N];

struct Node {
    int l, r;
    int s;
} tr[N * 21];
int root[N], tot;

int build(int l, int r) {
    int p = ++ tot;
    int mid = (l + r) >> 1;
    if (l < r) {
        tr[p].l = build(l, mid);
        tr[p].r = build(mid + 1, r);
    }
    tr[p].s = 0;
    return p;
}

int add(int p, int l, int r, int x) {
    int q = ++tot;
    int mid = (l + r) >> 1;
    tr[q] = tr[p];
    tr[q].s ++;
    if (l < r) {
        if (x <= mid) {
            tr[q].l = add(tr[p].l, l, mid, x);
        } else {
            tr[q].r = add(tr[p].r, mid + 1, r, x);
        }
    }
    return q;
}

int query(int p, int q, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return tr[q].s - tr[p].s;
    
    int mid = (l + r) >> 1;
    int res = 0;
    if (ql <= mid) res += query(tr[p].l, tr[q].l, l, mid, ql, qr);
    if (qr > mid) res += query(tr[p].r, tr[q].r, mid + 1, r, ql, qr);
    return res;
}

void add(int u, int v) {
    e[++ idx] = v;
    ne[idx] = h[u];
    h[u] = idx;
}

int dfn[N], rnk[N], low[N], fa[N][21], tim;

void dfs(int u, int father) {
    dfn[u] = ++ tim;
    rnk[tim] = u;
    
    fa[u][0] = father;
    for (int i = 1; i <= 20; ++ i) {
        fa[u][i] = fa[fa[u][i - 1]][i - 1];
    }
    
    for (int i = h[u]; i; i = ne[i]) {
        int v = e[i];
        if (v == father) continue ;
        dfs(v, u);
    }
    low[u] = tim;
}

vector<int> ve;

void solve() {
    int n;
    cin >> n;
    for (int i = 1; i < n; ++ i) {
        int u, v;
        cin >> u >> v;
        add(u, v);
        add(v, u);
    }
    
    for (int i = 1; i <= n; ++ i) {
        cin >> w[i];
        ve.push_back(w[i]);
    }
    
    //求dfs序
    dfs(1, 1);
    
    sort(ve.begin(), ve.end());
    ve.erase(unique(ve.begin(), ve.end()), ve.end());
    
    for (int i = 1; i <= n; ++ i) {
        w[i] = lower_bound(ve.begin(), ve.end(), w[i]) - ve.begin() + 1;
    }
    
    int len = ve.size();
    
    //构建主席树
    root[0] = build(1, len);
    for (int i = 1; i <= n; ++ i) {
        root[i] = add(root[i - 1], 1, len, w[rnk[i]]);
    }
    
    int q;
    cin >> q;
    while (q -- ) {
        int x, l, r;
        cin >> x >> l >> r;
        
        l = lower_bound(ve.begin(), ve.end(), l) - ve.begin() + 1;
        r = upper_bound(ve.begin(), ve.end(), r) - ve.begin();
        
        if (l <= w[x] && w[x] <= r) {
            
            for (int i = 20; i >= 0; -- i) {
                if (w[fa[x][i]] <= r) x = fa[x][i];
            }
            int ql = dfn[x] - 1;
            int qr = low[x];
            cout << query(root[ql], root[qr], 1, len, l, r) << "\n";
        } else {
            cout << 0 << "\n";
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    
    solve();
    
    return 0;
}
```



## 可持续化Trie树

> 例题：最大异或和

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 6e5 + 10;
const int M = 25 * N;

int n, m;
int s[N];
int tr[M][2], cnt[M];
int root[N], idx;

void add(int k, int p, int q) {
    cnt[q] = k;
    for (int i = 23; i >= 0; -- i) {
        int u = (s[k] >> i) & 1;
        if (p) {
            tr[q][u ^ 1] = tr[p][u ^ 1];
        }
        tr[q][u] = ++ idx;
        cnt[tr[q][u]] = k;
        p = tr[p][u];
        q = tr[q][u];
    }
}

int query(int p, int l, int c) {
    for (int i = 23; i >= 0; -- i) {
        int u = (c >> i) & 1;
        if (cnt[tr[p][u ^ 1]] >= l) {
            u ^= 1;
        }
        p = tr[p][u];
    }
    
    return c ^ s[cnt[p]];
}

void solve() {
    cnt[0] = -1;
    root[0] = ++ idx;
    add(0, 0, root[0]);
    
    cin >> n >> m;
    for (int i = 1; i <= n; ++ i) {
        cin >> s[i];
        s[i] ^= s[i - 1];
        
        root[i] = ++ idx;
        add(i, root[i - 1], root[i]);
    }
    
    while (m -- ) {
        char op;
        int l, r, x;
        cin >> op;
        if (op == 'A') {
            cin >> x;
            n ++;
            root[n] = ++ idx;
            s[n] = s[n - 1] ^ x;
            add(n, root[n - 1], root[n]);
        } else {
            cin >> l >> r >> x;
            cout << query(root[r - 1], l - 1, s[n] ^ x) << "\n";
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);
    
    solve();
    
    return 0;
}
```



## 并查集



### 带权并查集

> 可以用来维护点对之间的关系



#### 食物链

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



#### 奇偶游戏

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



## 线段树

> 可以用来维护区间问题



### 区间GCD

> 例题：[Problem - D - Codeforces](https://codeforces.com/contest/1549/problem/D)

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long LL;

const int N = 3e5 + 10;

LL w[N];
struct Node {
    int l, r;
    LL sum, d;
}; 

struct SegmentTree{
    Node tr[N << 2];
    void pushup(Node &u, Node &l, Node &r) {
        u.sum = l.sum + r.sum;
        u.d = gcd(l.d, r.d);
    }
    void pushup(int u) {
        pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
    }
    void build(int u, int l, int r) {
        if (l == r) {
            LL b = w[r] - w[r - 1];
            tr[u] = {l, r, b, b};
        } else {
            tr[u] = {l, r};
            int mid = l + r >> 1;
            build(u << 1, l, mid); build(u << 1 | 1, mid + 1, r);
            pushup(u);
        }
    }
    void modify(int u, int x, LL v) {
        if (tr[u].l == x && tr[u].r == x) {
            LL b = tr[u].sum + v;
            tr[u] = {x, x, b, b};
        } else {
            int mid = tr[u].l + tr[u].r >> 1;
            if (x <= mid) modify(u << 1, x, v);
            else modify(u << 1 | 1, x, v);
            pushup(u);
        }
    }
    Node query(int u, int l, int r) {
        if (l <= tr[u].l && tr[u].r <= r) return tr[u];
        else {
            int mid = tr[u].l + tr[u].r >> 1;
            if (r <= mid) return query(u << 1, l, r);
            else if (l > mid) return query(u << 1 | 1, l, r);
            else {
                auto left = query(u << 1, l, r);
                auto right = query(u << 1 | 1, l, r);
                Node res;
                pushup(res, left, right);
                return res;
            }
        }
    }
} seg;

LL query(int l, int r) {
    auto left = seg.query(1, 1, l);
    Node right({0, 0, 0, 0});
    if (l + 1 <= r) right = seg.query(1, l + 1, r);
    return abs(gcd(left.sum, right.d));
}

void solve() {
    int n;
    scanf("%d", &n);
    vector<LL> a(n + 1);
    for (int i = 1; i <= n; i++) {
        scanf("%lld", &a[i]);
    }
    if (n == 1) {
        printf("1\n");
        return ;
    }
    for (int i = 1; i < n; i++) {
        w[i] = a[i + 1] - a[i];
    }
    n--;
    seg.build(1, 1, n);

    int ans = 0;
    for (int l = 1, r = 1; r <= n; r++) {
        while (l <= r && query(l, r) == 1) l ++;
        ans = max(ans, r - l + 1);
    }

    printf("%d\n", ans + 1);
} 

int main() {
    int t;
    scanf("%d", &t);

    while (t--) {
        solve();
    } 

    return 0;
}
```



### 区间最大连续子段和

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long LL;

const int N = 5e5 + 10;

int n, m;
LL w[N];

struct Segment_Tree {
    struct Node {
        int l, r;
        LL sum, lmax, rmax, tmax;
    } tr[N * 4];
    
    void pushup(Node &u, Node &l, Node &r) {
        u.sum = l.sum + r.sum;
        u.lmax = max(l.lmax, l.sum + r.lmax);
        u.rmax = max(r.rmax, r.sum + l.rmax);
        u.tmax = max(max(l.tmax, r.tmax), l.rmax + r.lmax);
    }
    
    void pushup(int u) {
        pushup(tr[u], tr[u << 1], tr[u << 1 | 1]);
    }
    
    void build(int u, int l, int r) {
        if (l == r) tr[u] = {l, r, w[r], w[r], w[r], w[r]};
        else {
            tr[u] = {l, r};
            int mid = l + r >> 1;
            build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
            pushup(u);
        }
    }

    
    void modify(int u, int x, int v) {
        if (tr[u].l == x && tr[u].r == x) tr[u] = {x, x, v, v, v, v};
        else {
            int mid = tr[u].l + tr[u].r >> 1;
            if (x <= mid) modify(u << 1, x, v);
            else modify(u << 1 | 1, x, v);
            pushup(u);
        }
    }
    
    Node query(int u, int l, int r) {
        if (tr[u].l >= l && tr[u].r <= r) return tr[u];
        else {
            int mid = tr[u].l + tr[u].r >> 1;
            if (r <= mid) return query(u << 1, l, r);
            else if (l > mid) return query(u << 1 | 1, l, r);
            else {
                auto left = query(u << 1, l, r);
                auto right = query(u << 1 | 1, l, r);
                Node res;
                pushup(res, left, right);
                return res;
            }
        }
    }
} seg;

void solve() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) scanf("%lld", &w[i]);
    
    seg.build(1, 1, n);
    
    while (m -- ) {
        int op, x, y;
        scanf("%d%d%d", &op, &x, &y);
        if (op == 1) {
            if (x > y) swap(x, y);
            printf("%lld\n", seg.query(1, x, y).tmax);
        } else seg.modify(1, x, y);
    }
}

int main() {
    solve();
    
    return 0;
}
```



## ST表

> 可以维护静态的区间可重复贡献问题



### 区间最值

```cpp
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e5 + 10;
const int M = 30;

int n, m;
int logn[N], f[N][M];

int main () {
    scanf ("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) scanf ("%d", &f[i][0]);
    for (int i = 2; i <= n; ++ i) logn[i] = logn[i / 2] + 1;

    for (int j = 1; j < M; ++ j)
        for (int i = 1; i + (1 << j) - 1 <= n; ++ i) 
            f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);


    while (m --) {
        int l, r;
        scanf ("%d%d", &l, &r);
        int k = logn[r - l + 1];
        printf ("%d\n", max(f[l][k], f[r - (1 << k) + 1][k])); 
    }

    return 0;
}
```



### 区间GCD

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long LL;

constexpr int N = 2e5;

int logn[N];

int query(int l, int r) {
    int k = logn[r - l + 1];
    return gcd(f[l][k], f[r - (1 << k) + 1][k]);
}

void solve() {
    int n;
    cin >> n;

    vector<LL> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    n--;
    vector<vector<LL>> f(n + 1, vector<LL>(21, 0));
    for (int i = 0; i < n; i++) {
        f[i][0] = a[i + 1] - a[i];
    }

    for (int j = 1; j < 21; j++) {
        for (int i = 0; i + (1 << j) < n; i++) {
            f[i][j] = gcd(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
        }
    }

    int ans = 0;
    for (int l = 0, r = 0; r < n; r++) {
        while (l <= r && query(l, r) == 1) l ++;
        ans = max(ans, r - l + 1);
    }

    ans++;
    cout << ans << "\n";
} 

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;

    for (int i = 2; i < N; i ++) {
        logn[i] = logn[i / 2] + 1;
    }

    while (t--) {
        solve();
    } 

    return 0;
}
```



## 单调队列

> 滑动窗口

```cpp
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e6 + 10;

int n, k;
int a[N], q[N];

int main () {
    scanf ("%d%d", &n, &k);
    for (int i = 1; i <= n; ++ i) scanf("%d", &a[i]);

    int hh = 0, tt = -1;
    for (int i = 1; i <= n; ++ i) {
        if (hh <= tt && i - k >= q[hh]) hh ++;

        while (hh <= tt && a[q[tt]] >= a[i]) tt --;
        q[++ tt] = i;

        if (i - k >= 0) printf ("%d ", a[q[hh]]);
    }
    puts("");
    hh = 0; tt = -1;
    for (int i = 1; i <= n; ++ i) {
        if (hh <= tt && i - k >= q[hh]) hh ++;

        while (hh <= tt && a[q[tt]] <= a[i]) tt --;
        q[++ tt] = i;

        if (i - k >= 0) printf ("%d ", a[q[hh]]);
    }

    return 0;
}
```



## 二叉搜索树



### 无旋Treap

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;
const int INF = 1e8;

struct fhq_treap {
    struct Node {
    	int l, r;
    	int key, val;
    	int size;
    } tr[N];
    int root, idx;
    
    fhq_treap() {
        add (-INF); add (INF);
    }
    
    void pushup (int p) { 
    	tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + 1; 
    }
    int New (int key) {
    	tr[++ idx].key = key;
    	tr[idx].val = rand ();
    	tr[idx].size = 1;
    	return idx;
    }
    void split (int p, int key, int &x, int &y) {
    	if (!p) { x = y = 0; return ; } //到底了
    	else if (key >= tr[p].key) { //如果值比点大了，就要往右边找 
    		x = p; //给x 
    		split (tr[p].r, key, tr[p].r, y); //往右边找 
    	} else {
    		y = p; //给y 
    		split (tr[p].l, key, x, tr[p].l); //往左边找 
    	}
    	pushup (p);
    } 
    int merge (int x, int y) {
    	if (!x || !y) return x + y;
    	if (tr[x].val <= tr[y].val) { //满足小根堆 
    		tr[x].r = merge (tr[x].r, y);
    		pushup (x);
    		return x;
    	} else {
    		tr[y].l = merge (x, tr[y].l);
    		pushup (y);
    		return y;
    	}
    }
    
    void add (int key) {
    	int x, y;
    	split (root, key, x, y);
    	root = merge(merge(x, New (key)), y);
    }
    void del (int key) {
    	int x, y, z;
    	split (root, key, x, z);
    	split (x, key - 1, x, y);
    	y = merge (tr[y].l, tr[y].r);
    	root = merge (merge (x, y), z);
    }
    
    int rnk (int key) { //通过key找到对应的排名 
    	int x, y;
    	split (root, key - 1, x, y);
    	int rank = tr[x].size + 1;
    	root = merge (x, y);
    	return rank - 1;
    } 
    int kth(int x, int k) {
        while(1) {
            int ls = tr[x].l, rs = tr[x].r;
            if (k <= tr[ls].size) x = ls;
            else if (k > tr[ls].size + 1) k -= tr[ls].size + 1, x = rs;
            else return x;
        }
    }
    int pre (int key) {
    	int x, y;
    	split (root, key - 1, x, y);
    	int ans = tr[kth (x, tr[x].size)].key;
    	root = merge (x, y);
    	return ans;
    }
    int nxt (int key) {
    	int x, y;
    	split (root, key, x, y);
    	int ans = tr[kth(y, 1)].key;
    	root = merge (x, y);
    	return ans;
    } 
    int kth(int k) {
        return tr[kth(root, k + 1)].key;
    }
} treap;


void solve() {
    int n;
    scanf("%d", &n);
    while (n -- ) {
        int opt, x;
        scanf("%d%d", &opt, &x);
        if (opt == 1) treap.add(x);
        else if (opt == 2) treap.del(x);
        else if (opt == 3) printf("%d\n", treap.rnk(x));
        else if (opt == 4) printf("%d\n", treap.kth(x));
        else if (opt == 5) printf("%d\n", treap.pre(x));
        else printf("%d\n", treap.nxt(x));
    }
}

int main() {
    solve();
    return 0;
}
```



### Splay

> 伸展树，可以做到区间翻转

```cpp
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e5 + 10;
const int INF = 1e9 + 7;

struct Node {
    int s[2];
    int p, v;
    int size;
    void init(int _v, int _p) {
        v = _v, p = _p;
        size = 1;
    }
} tr[N];
int root, idx;

void pushup(int x) {
    tr[x].size = tr[tr[x].s[0]].size + tr[tr[x].s[1]].size + 1;
}
void rotate(int x) {
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;
    tr[z].s[tr[z].s[1] == y] = x; tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1]; tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y; tr[y].p = x;
    pushup(y); pushup(x);
}

void splay(int x, int k) {
    while (tr[x].p != k) {
        int y = tr[x].p, z = tr[y].p;
        if (z != k) {
            if ((tr[z].s[1] == y) ^ (tr[y].s[1] == x)) rotate(x);
            else rotate(y);
        }
        rotate(x);
    }
    if (!k) root = x;
}

int insert(int v) {
    int u = root, p = 0;
    while (u) p = u, u = tr[u].s[v > tr[u].v];
    u = ++ idx;
    if (p) tr[p].s[v > tr[p].v] = u;
    tr[u].init(v, p);
    splay(u, 0);
    return u;
}

int kth (int k) {
    int u = root;
    while (1) {
        if (k <= tr[tr[u].s[0]].size) u = tr[u].s[0];
        else if (k == tr[tr[u].s[0]].size + 1) return u;
        else k -= tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
    }
    return -1;
}

int getv (int v) {
    int u = root, p = 0;
    while (u) {
        if (v <= tr[u].v) p = u, u = tr[u].s[0];
        else u = tr[u].s[1];
    }
    return p;
}

int main () {
    int n, m;
    cin >> n >> m;
    int L = insert(-INF), R = insert(INF);
    
    int cnt = 0, d = 0;
    while (n -- ) {
        char opt; int k;
        cin >> opt >> k;
        if (opt == 'I') {
            if (k >= m) {
                k -= d;
                insert(k);
                cnt ++;
            }
        } 
        else if (opt == 'A') d += k;
        else if (opt == 'S') {
            d -= k;
            R = getv(m - d);
            splay(R, 0); splay(L, R);
            tr[L].s[1] = 0;
            pushup(L); pushup(R);
        } 
        else if (opt == 'F') {
            if (tr[root].size - 2 < k) puts("-1");
            else printf ("%d\n", tr[kth(tr[root].size - k)].v + d);
        }
    }
    printf ("%d", cnt - tr[root].size + 2);
    
    return 0;
}
```



## 树状数组



### 单点修改，区间查询

```cpp
inline int lowbit(int x) {
    return x & (-x);
}

void add(int x, int k) {
    for (; x <= n; x += lowbit(x))
        tr[x] += k;
}

int query(int x) {
    int ans = 0;
    for (; x; x -= lowbit(x))
        ans += tr[x];
       return ans;
}
```






# 图论



## 拓扑排序



### 求有向图的拓扑序

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



### 拓扑排序去非环点

> 例题：[Link with Limit](https://acm.hdu.edu.cn/showproblem.php?pid=7050)

```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    int n;
    scanf("%d", &n);
    vector<int> in(n + 1, 0), f(n + 1, 0);
    for (int i = 1; i <= n; ++ i) {
        scanf("%d", &f[i]);
        in[f[i]] ++;
    }
    
    queue<int> que;
    for (int i = 1; i <= n; ++ i)
        if (!in[i]) que.push(i);
        
    while (!que.empty()) {
        int u = que.front();
        que.pop();
        
        int v = f[u];
        
        in[v] --;
        if (in[v] == 0) que.push(v);
    }
    
    int p = -1, q = -1;
    
    for(int i = 1; i <= n; ++ i) {
        if (in[i] == 0) continue ;
        int sum = 0, tot = 0;
        for (int j = i; in[j]; j = f[j]) {
            in[j] = 0;
            sum += j;
            tot ++;
        }
        if (p == -1 && q == -1) {
            p = sum, q = tot;
        } else if (p * tot != sum * q) {
            puts("NO");
            return ;
        }
    }
    
    puts("YES");
}

int main() {
    int T;
    scanf("%d", &T);
    while (T --) {
        solve();
    }
    
    return 0;
}
```



## 最短路



### dijkstra算法

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



### SPFA算法

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



### Johnson全源最短路

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



## 最小生成树



### Prim算法

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



### kruscal算法

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



## 二分图



### 二分图判定



#### 染色法

```cpp
#include <bits/stdc++.h>
using namespace std;

template <typename T>
inline void read(T &x) {
    x = 0; int f = 1; char ch = getchar();
    while (!isdigit(ch)) { if (ch == '-') f = -1; ch = getchar(); }
    while (isdigit(ch)) { x = (x << 3) + (x << 1) + (ch & 15); ch = getchar(); }
    x *= f;
}

const int N = 1e5 + 10;

int n, m;
vector<int> G[N];
int color[N];

bool dfs(int u, int c) {
    color[u] = c;
    for (auto v : G[u]) {
        if (!color[v]) {
            if (!dfs(v, 3 - c))
                return false;
        } else {
            if (color[v] == c) return false;
        }
    }
    return true;
}

int main () {
    read(n); read(m);
    while (m --) {
        int u, v;
        read(u); read(v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    
    bool flag = true;
    for (int i = 1; i <= n; ++ i) 
        if (!color[i]) 
            if (!dfs(i, 1)) {
                flag = false;
                break;
            }
    
    if (flag) puts("Yes");
    else puts("No");
    
    return 0;
}
```



### 最大匹配问题



#### KM算法

> 匈牙利算法

```cpp
#include <bits/stdc++.h>
using namespace std;

template <typename T>
inline void read(T &x){
	x = 0; int f = 1; char ch = getchar();
	while (!isdigit (ch)) { if (ch == '-') f = -1; ch = getchar(); }
	while (isdigit(ch)) { x = (x << 3) + (x << 1) + (ch & 15); ch = getchar(); }
	x *= f;
}

const int N = 500 + 10, M = 2e5 + 10;

int n1, n2, m;
vector<int> G[N];
bool vis[N];
int match[N];

int find(int u) {
	for (auto v : G[u]) {
	    if (!vis[v]) {
			vis[v] = true;
			if (match[v] == 0 || find(match[v])) {
				match[v] = u;
				return true;
			}
		}
	}
	return false;
}

int main () {
	read(n1), read(n2), read(m);
	while (m --) {
		int u, v;
		read(u), read(v);
		G[u].push_back(v);
	}
	
	int res = 0;
	for (int i = 1; i <= n1; ++ i) {
		memset(vis, false, sizeof vis);
		if (find(i)) res ++;
	}
	
	printf ("%d", res);
	
	return 0;
}
```



## 树上问题



### 最近公共祖先



#### 倍增法

```cpp
#include <bits/stdc++.h>
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



#### Tarjan离线算法

```cpp
#include <bits/stdc++.h>
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



#### 树剖

```cpp
#pragma GCC optimize(2)
#include <bits/stdc++.h>
using namespace std;

const int N = 4e4 + 10;

int n, m;
vector<int> G[N];
int dfn[N], siz[N], fa[N], son[N], top[N], dep[N], idx;

void dfs1(int u, int father, int depth) {
    dep[u] = depth; fa[u] = father; siz[u] = 1;
    for (auto &v : G[u]) {
        if (v == father) continue;
        dfs1 (v, u, depth + 1);
        siz[u] += siz[v];
        if (siz[son[u]] < siz[v]) son[u] = v;
    }
}
void dfs2(int u, int t) {
    dfn[u] = ++ idx;
    top[u] = t;
    if (!son[u]) return ;
    dfs2(son[u], t);
    for (auto &v : G[u]) {
        if (v == fa[u] || v == son[u]) continue;
        dfs2(v, v);
    }
}

int lca(int a, int b) {
    while (top[a] != top[b]) {
        if (dep[top[a]] < dep[top[b]]) swap(a, b);
        a = fa[top[a]];
    }
    return dep[a] < dep[b] ? a : b;
}

int main() {
    scanf("%d", &n);
    int root = -1;
    for (int i = 1; i <= n; ++ i) {
        int a, b;
        scanf("%d%d", &a, &b);
        if (b == -1) {
            root = a;
        } else {
            G[a].push_back(b);
            G[b].push_back(a);
        }
    }
    
    dfs1 (root, root, 1);
    dfs2 (root, root);
    
    scanf("%d", &m);
    while (m -- ) {
        int a, b;
        scanf("%d%d", &a, &b);
        int f = lca(a, b);
        if (a == f) puts("1");
        else if (b == f) puts("2");
        else puts("0");
    }
    
    return 0;
}
```







# 数学



## 最大公约数

```cpp
int gcd (int a , int b) {
    return b ? gcd (b , a % b) : a;
}
```



## 乘法逆元



### 费马小定理

```cpp
typedef long long LL;

LL qmi (LL a , LL b , LL p) {
    LL res = 1 % p;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return res % p;
}

LL inv(LL n, LL p) {
    return qmi(n, p - 2, p);
}
```



### 拓展欧几里得

```cpp
int exgcd (int a, int b, int &x, int &y) {
    if (!b) {
        x = 1; y = 0;
        return a;
    }
    int g = exgcd (b, a % b, y, x);
    y -= a / b * x;
    return g;
}
int inv(int n, int p) {
    int x, y;
       int g = exgcd(n, p, x, y);
    if(g == 1) return (x % p + p) % p;
    else return -1;
}
```



## 扩展欧几里得

> 可以用来求解二元一次方程

```cpp
int exgcd (int a, int b, int &x, int &y) {
    if (!b) {
        x = 1; y = 0;
        return a;
    }
    int g = exgcd (b, a % b, y, x);
    y -= a / b * x;
    return g;
}
```



## 素数



### 素数判定

```cpp
bool is_prime(int x) 
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}
```



### 分解质因子

```cpp
void divide (int x) {
    for (int i = 2 ; i <= x / i ; ++ i) {
        if (x % i == 0) {
            int t = 0;
            while (x % i == 0) x /= i , t ++;
            printf ("%d %d\n" , i , t);
        }
    }
    if (x > 1) printf ("%d %d\n" , x , 1);
}
```



### 欧拉筛素数

```cpp
    const int N = 1000000 + 10;

int p[N] , v[N];

bool get_prime (int n) {
    p[0] = 0;
    v[1] = 1;
    for (int i = 2 ; i <= n ; ++ i) {
        if (v[i] == 0) p[++p[0]] = i;

        for (int j = 1 ; j <= p[0] && i * p[j] <= n ; ++ j) {
            v[i * p[j]] = 1;
            if (i % p[j] == 0) break;
        }
    }
}
```



# 计算几何



## 二维几何

> 来自kuangbin

```cpp
// `计算几何模板`
const double eps = 1e-8;
const double inf = 1e20;
const double pi = acos(-1.0);
const int maxp = 1010;
//`Compares a double to zero`
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
//square of a double
inline double sqr(double x){return x*x;}
/*
 * Point
 * Point()               - Empty constructor
 * Point(double _x,double _y)  - constructor
 * input()             - double input
 * output()            - %.2f output
 * operator ==         - compares x and y
 * operator <          - compares first by x, then by y
 * operator -          - return new Point after subtracting curresponging x and y
 * operator ^          - cross product of 2d points
 * operator *          - dot product
 * len()               - gives length from origin
 * len2()              - gives square of length from origin
 * distance(Point p)   - gives distance from p
 * operator + Point b  - returns new Point after adding curresponging x and y
 * operator * double k - returns new Point after multiplieing x and y by k
 * operator / double k - returns new Point after divideing x and y by k
 * rad(Point a,Point b)- returns the angle of Point a and Point b from this Point
 * trunc(double r)     - return Point that if truncated the distance from center to r
 * rotleft()           - returns 90 degree ccw rotated point
 * rotright()          - returns 90 degree cw rotated point
 * rotate(Point p,double angle) - returns Point after rotateing the Point centering at p by angle radian ccw
 */
struct Point{
	double x,y;
	Point(){}
	Point(double _x,double _y){
		x = _x;
		y = _y;
	}
	void input(){
		scanf("%lf%lf",&x,&y);
	}
	void output(){
		printf("%.2f %.2f\n",x,y);
	}
	bool operator == (Point b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0;
	}
	bool operator < (Point b)const{
		return sgn(x-b.x)== 0?sgn(y-b.y)<0:x<b.x;
	}
	Point operator -(const Point &b)const{
		return Point(x-b.x,y-b.y);
	}
	//叉积
	double operator ^(const Point &b)const{
		return x*b.y - y*b.x;
	}
	//点积
	double operator *(const Point &b)const{
		return x*b.x + y*b.y;
	}
	//返回长度
	double len(){
		return hypot(x,y);//库函数
	}
	//返回长度的平方
	double len2(){
		return x*x + y*y;
	}
	//返回两点的距离
	double distance(Point p){
		return hypot(x-p.x,y-p.y);
	}
	Point operator +(const Point &b)const{
		return Point(x+b.x,y+b.y);
	}
	Point operator *(const double &k)const{
		return Point(x*k,y*k);
	}
	Point operator /(const double &k)const{
		return Point(x/k,y/k);
	}
	//`计算pa  和  pb 的夹角`
	//`就是求这个点看a,b 所成的夹角`
	//`测试 LightOJ1203`
	double rad(Point a,Point b){
		Point p = *this;
		return fabs(atan2( fabs((a-p)^(b-p)),(a-p)*(b-p) ));
	}
	//`化为长度为r的向量`
	Point trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point(x*r,y*r);
	}
	//`逆时针旋转90度`
	Point rotleft(){
		return Point(-y,x);
	}
	//`顺时针旋转90度`
	Point rotright(){
		return Point(y,-x);
	}
	//`绕着p点逆时针旋转angle`
	Point rotate(Point p,double angle){
		Point v = (*this) - p;
		double c = cos(angle), s = sin(angle);
		return Point(p.x + v.x*c - v.y*s,p.y + v.x*s + v.y*c);
	}
};
/*
 * Stores two points
 * Line()                         - Empty constructor
 * Line(Point _s,Point _e)        - Line through _s and _e
 * operator ==                    - checks if two points are same
 * Line(Point p,double angle)     - one end p , another end at angle degree
 * Line(double a,double b,double c) - Line of equation ax + by + c = 0
 * input()                        - inputs s and e
 * adjust()                       - orders in such a way that s < e
 * length()                       - distance of se
 * angle()                        - return 0 <= angle < pi
 * relation(Point p)              - 3 if point is on line
 *                                  1 if point on the left of line
 *                                  2 if point on the right of line
 * pointonseg(double p)           - return true if point on segment
 * parallel(Line v)               - return true if they are parallel
 * segcrossseg(Line v)            - returns 0 if does not intersect
 *                                  returns 1 if non-standard intersection
 *                                  returns 2 if intersects
 * linecrossseg(Line v)           - line and seg
 * linecrossline(Line v)          - 0 if parallel
 *                                  1 if coincides
 *                                  2 if intersects
 * crosspoint(Line v)             - returns intersection point
 * dispointtoline(Point p)        - distance from point p to the line
 * dispointtoseg(Point p)         - distance from p to the segment
 * dissegtoseg(Line v)            - distance of two segment
 * lineprog(Point p)              - returns projected point p on se line
 * symmetrypoint(Point p)         - returns reflection point of p over se
 *
 */
struct Line{
	Point s,e;
	Line(){}
	Line(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	bool operator ==(Line v){
		return (s == v.s)&&(e == v.e);
	}
	//`根据一个点和倾斜角angle确定直线,0<=angle<pi`
	Line(Point p,double angle){
		s = p;
		if(sgn(angle-pi/2) == 0){
			e = (s + Point(0,1));
		}
		else{
			e = (s + Point(1,tan(angle)));
		}
	}
	//ax+by+c=0
	Line(double a,double b,double c){
		if(sgn(a) == 0){
			s = Point(0,-c/b);
			e = Point(1,-c/b);
		}
		else if(sgn(b) == 0){
			s = Point(-c/a,0);
			e = Point(-c/a,1);
		}
		else{
			s = Point(0,-c/b);
			e = Point(1,(-c-a)/b);
		}
	}
	void input(){
		s.input();
		e.input();
	}
	void adjust(){
		if(e < s)swap(s,e);
	}
	//求线段长度
	double length(){
		return s.distance(e);
	}
	//`返回直线倾斜角 0<=angle<pi`
	double angle(){
		double k = atan2(e.y-s.y,e.x-s.x);
		if(sgn(k) < 0)k += pi;
		if(sgn(k-pi) == 0)k -= pi;
		return k;
	}
	//`点和直线关系`
	//`1  在左侧`
	//`2  在右侧`
	//`3  在直线上`
	int relation(Point p){
		int c = sgn((p-s)^(e-s));
		if(c < 0)return 1;
		else if(c > 0)return 2;
		else return 3;
	}
	// 点在线段上的判断
	bool pointonseg(Point p){
		return sgn((p-s)^(e-s)) == 0 && sgn((p-s)*(p-e)) <= 0;
	}
	//`两向量平行(对应直线平行或重合)`
	bool parallel(Line v){
		return sgn((e-s)^(v.e-v.s)) == 0;
	}
	//`两线段相交判断`
	//`2 规范相交`
	//`1 非规范相交`
	//`0 不相交`
	int segcrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		int d3 = sgn((v.e-v.s)^(s-v.s));
		int d4 = sgn((v.e-v.s)^(e-v.s));
		if( (d1^d2)==-2 && (d3^d4)==-2 )return 2;
		return (d1==0 && sgn((v.s-s)*(v.s-e))<=0) ||
			(d2==0 && sgn((v.e-s)*(v.e-e))<=0) ||
			(d3==0 && sgn((s-v.s)*(s-v.e))<=0) ||
			(d4==0 && sgn((e-v.s)*(e-v.e))<=0);
	}
	//`直线和线段相交判断`
	//`-*this line   -v seg`
	//`2 规范相交`
	//`1 非规范相交`
	//`0 不相交`
	int linecrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		if((d1^d2)==-2) return 2;
		return (d1==0||d2==0);
	}
	//`两直线关系`
	//`0 平行`
	//`1 重合`
	//`2 相交`
	int linecrossline(Line v){
		if((*this).parallel(v))
			return v.relation(s)==3;
		return 2;
	}
	//`求两直线的交点`
	//`要保证两直线不平行或重合`
	Point crosspoint(Line v){
		double a1 = (v.e-v.s)^(s-v.s);
		double a2 = (v.e-v.s)^(e-v.s);
		return Point((s.x*a2-e.x*a1)/(a2-a1),(s.y*a2-e.y*a1)/(a2-a1));
	}
	//点到直线的距离
	double dispointtoline(Point p){
		return fabs((p-s)^(e-s))/length();
	}
	//点到线段的距离
	double dispointtoseg(Point p){
		if(sgn((p-s)*(e-s))<0 || sgn((p-e)*(s-e))<0)
			return min(p.distance(s),p.distance(e));
		return dispointtoline(p);
	}
	//`返回线段到线段的距离`
	//`前提是两线段不相交，相交距离就是0了`
	double dissegtoseg(Line v){
		return min(min(dispointtoseg(v.s),dispointtoseg(v.e)),min(v.dispointtoseg(s),v.dispointtoseg(e)));
	}
	//`返回点p在直线上的投影`
	Point lineprog(Point p){
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`返回点p关于直线的对称点`
	Point symmetrypoint(Point p){
		Point q = lineprog(p);
		return Point(2*q.x-p.x,2*q.y-p.y);
	}
};
//圆
struct circle{
	Point p;//圆心
	double r;//半径
	circle(){}
	circle(Point _p,double _r){
		p = _p;
		r = _r;
	}
	circle(double x,double y,double _r){
		p = Point(x,y);
		r = _r;
	}
	//`三角形的外接圆`
	//`需要Point的+ /  rotate()  以及Line的crosspoint()`
	//`利用两条边的中垂线得到圆心`
	//`测试：UVA12304`
	circle(Point a,Point b,Point c){
		Line u = Line((a+b)/2,((a+b)/2)+((b-a).rotleft()));
		Line v = Line((b+c)/2,((b+c)/2)+((c-b).rotleft()));
		p = u.crosspoint(v);
		r = p.distance(a);
	}
	//`三角形的内切圆`
	//`参数bool t没有作用，只是为了和上面外接圆函数区别`
	//`测试：UVA12304`
	circle(Point a,Point b,Point c,bool t){
		Line u,v;
		double m = atan2(b.y-a.y,b.x-a.x), n = atan2(c.y-a.y,c.x-a.x);
		u.s = a;
		u.e = u.s + Point(cos((n+m)/2),sin((n+m)/2));
		v.s = b;
		m = atan2(a.y-b.y,a.x-b.x) , n = atan2(c.y-b.y,c.x-b.x);
		v.e = v.s + Point(cos((n+m)/2),sin((n+m)/2));
		p = u.crosspoint(v);
		r = Line(a,b).dispointtoseg(p);
	}
	//输入
	void input(){
		p.input();
		scanf("%lf",&r);
	}
	//输出
	void output(){
		printf("%.2lf %.2lf %.2lf\n",p.x,p.y,r);
	}
	bool operator == (circle v){
		return (p==v.p) && sgn(r-v.r)==0;
	}
	bool operator < (circle v)const{
		return ((p<v.p)||((p==v.p)&&sgn(r-v.r)<0));
	}
	//面积
	double area(){
		return pi*r*r;
	}
	//周长
	double circumference(){
		return 2*pi*r;
	}
	//`点和圆的关系`
	//`0 圆外`
	//`1 圆上`
	//`2 圆内`
	int relation(Point b){
		double dst = b.distance(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r)==0)return 1;
		return 0;
	}
	//`线段和圆的关系`
	//`比较的是圆心到线段的距离和半径的关系`
	int relationseg(Line v){
		double dst = v.dispointtoseg(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`直线和圆的关系`
	//`比较的是圆心到直线的距离和半径的关系`
	int relationline(Line v){
		double dst = v.dispointtoline(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`两圆的关系`
	//`5 相离`
	//`4 外切`
	//`3 相交`
	//`2 内切`
	//`1 内含`
	//`需要Point的distance`
	//`测试：UVA12304`
	int relationcircle(circle v){
		double d = p.distance(v.p);
		if(sgn(d-r-v.r) > 0)return 5;
		if(sgn(d-r-v.r) == 0)return 4;
		double l = fabs(r-v.r);
		if(sgn(d-r-v.r)<0 && sgn(d-l)>0)return 3;
		if(sgn(d-l)==0)return 2;
		if(sgn(d-l)<0)return 1;
	}
	//`求两个圆的交点，返回0表示没有交点，返回1是一个交点，2是两个交点`
	//`需要relationcircle`
	//`测试：UVA12304`
	int pointcrosscircle(circle v,Point &p1,Point &p2){
		int rel = relationcircle(v);
		if(rel == 1 || rel == 5)return 0;
		double d = p.distance(v.p);
		double l = (d*d+r*r-v.r*v.r)/(2*d);
		double h = sqrt(r*r-l*l);
		Point tmp = p + (v.p-p).trunc(l);
		p1 = tmp + ((v.p-p).rotleft().trunc(h));
		p2 = tmp + ((v.p-p).rotright().trunc(h));
		if(rel == 2 || rel == 4)
			return 1;
		return 2;
	}
	//`求直线和圆的交点，返回交点个数`
	int pointcrossline(Line v,Point &p1,Point &p2){
		if(!(*this).relationline(v))return 0;
		Point a = v.lineprog(p);
		double d = v.dispointtoline(p);
		d = sqrt(r*r-d*d);
		if(sgn(d) == 0){
			p1 = a;
			p2 = a;
			return 1;
		}
		p1 = a + (v.e-v.s).trunc(d);
		p2 = a - (v.e-v.s).trunc(d);
		return 2;
	}
	//`得到过a,b两点，半径为r1的两个圆`
	int gercircle(Point a,Point b,double r1,circle &c1,circle &c2){
		circle x(a,r1),y(b,r1);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r;
		return t;
	}
	//`得到与直线u相切，过点q,半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(Line u,Point q,double r1,circle &c1,circle &c2){
		double dis = u.dispointtoline(q);
		if(sgn(dis-r1*2)>0)return 0;
		if(sgn(dis) == 0){
			c1.p = q + ((u.e-u.s).rotleft().trunc(r1));
			c2.p = q + ((u.e-u.s).rotright().trunc(r1));
			c1.r = c2.r = r1;
			return 2;
		}
		Line u1 = Line((u.s + (u.e-u.s).rotleft().trunc(r1)),(u.e + (u.e-u.s).rotleft().trunc(r1)));
		Line u2 = Line((u.s + (u.e-u.s).rotright().trunc(r1)),(u.e + (u.e-u.s).rotright().trunc(r1)));
		circle cc = circle(q,r1);
		Point p1,p2;
		if(!cc.pointcrossline(u1,p1,p2))cc.pointcrossline(u2,p1,p2);
		c1 = circle(p1,r1);
		if(p1 == p2){
			c2 = c1;
			return 1;
		}
		c2 = circle(p2,r1);
		return 2;
	}
	//`同时与直线u,v相切，半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(Line u,Line v,double r1,circle &c1,circle &c2,circle &c3,circle &c4){
		if(u.parallel(v))return 0;//两直线平行
		Line u1 = Line(u.s + (u.e-u.s).rotleft().trunc(r1),u.e + (u.e-u.s).rotleft().trunc(r1));
		Line u2 = Line(u.s + (u.e-u.s).rotright().trunc(r1),u.e + (u.e-u.s).rotright().trunc(r1));
		Line v1 = Line(v.s + (v.e-v.s).rotleft().trunc(r1),v.e + (v.e-v.s).rotleft().trunc(r1));
		Line v2 = Line(v.s + (v.e-v.s).rotright().trunc(r1),v.e + (v.e-v.s).rotright().trunc(r1));
		c1.r = c2.r = c3.r = c4.r = r1;
		c1.p = u1.crosspoint(v1);
		c2.p = u1.crosspoint(v2);
		c3.p = u2.crosspoint(v1);
		c4.p = u2.crosspoint(v2);
		return 4;
	}
	//`同时与不相交圆cx,cy相切，半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(circle cx,circle cy,double r1,circle &c1,circle &c2){
		circle x(cx.p,r1+cx.r),y(cy.p,r1+cy.r);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r1;
		return t;
	}

	//`过一点作圆的切线(先判断点和圆的关系)`
	//`测试：UVA12304`
	int tangentline(Point q,Line &u,Line &v){
		int x = relation(q);
		if(x == 2)return 0;
		if(x == 1){
			u = Line(q,q + (q-p).rotleft());
			v = u;
			return 1;
		}
		double d = p.distance(q);
		double l = r*r/d;
		double h = sqrt(r*r-l*l);
		u = Line(q,p + ((q-p).trunc(l) + (q-p).rotleft().trunc(h)));
		v = Line(q,p + ((q-p).trunc(l) + (q-p).rotright().trunc(h)));
		return 2;
	}
	//`求两圆相交的面积`
	double areacircle(circle v){
		int rel = relationcircle(v);
		if(rel >= 4)return 0.0;
		if(rel <= 2)return min(area(),v.area());
		double d = p.distance(v.p);
		double hf = (r+v.r+d)/2.0;
		double ss = 2*sqrt(hf*(hf-r)*(hf-v.r)*(hf-d));
		double a1 = acos((r*r+d*d-v.r*v.r)/(2.0*r*d));
		a1 = a1*r*r;
		double a2 = acos((v.r*v.r+d*d-r*r)/(2.0*v.r*d));
		a2 = a2*v.r*v.r;
		return a1+a2-ss;
	}
	//`求圆和三角形pab的相交面积`
	//`测试：POJ3675 HDU3982 HDU2892`
	double areatriangle(Point a,Point b){
		if(sgn((p-a)^(p-b)) == 0)return 0.0;
		Point q[5];
		int len = 0;
		q[len++] = a;
		Line l(a,b);
		Point p1,p2;
		if(pointcrossline(l,q[1],q[2])==2){
			if(sgn((a-q[1])*(b-q[1]))<0)q[len++] = q[1];
			if(sgn((a-q[2])*(b-q[2]))<0)q[len++] = q[2];
		}
		q[len++] = b;
		if(len == 4 && sgn((q[0]-q[1])*(q[2]-q[1]))>0)swap(q[1],q[2]);
		double res = 0;
		for(int i = 0;i < len-1;i++){
			if(relation(q[i])==0||relation(q[i+1])==0){
				double arg = p.rad(q[i],q[i+1]);
				res += r*r*arg/2.0;
			}
			else{
				res += fabs((q[i]-p)^(q[i+1]-p))/2.0;
			}
		}
		return res;
	}
};

/*
 * n,p  Line l for each side
 * input(int _n)                        - inputs _n size polygon
 * add(Point q)                         - adds a point at end of the list
 * getline()                            - populates line array
 * cmp                                  - comparision in convex_hull order
 * norm()                               - sorting in convex_hull order
 * getconvex(polygon &convex)           - returns convex hull in convex
 * Graham(polygon &convex)              - returns convex hull in convex
 * isconvex()                           - checks if convex
 * relationpoint(Point q)               - returns 3 if q is a vertex
 *                                                2 if on a side
 *                                                1 if inside
 *                                                0 if outside
 * convexcut(Line u,polygon &po)        - left side of u in po
 * gercircumference()                   - returns side length
 * getarea()                            - returns area
 * getdir()                             - returns 0 for cw, 1 for ccw
 * getbarycentre()                      - returns barycenter
 *
 */
struct polygon{
	int n;
	Point p[maxp];
	Line l[maxp];
	void input(int _n){
		n = _n;
		for(int i = 0;i < n;i++)
			p[i].input();
	}
	void add(Point q){
		p[n++] = q;
	}
	void getline(){
		for(int i = 0;i < n;i++){
			l[i] = Line(p[i],p[(i+1)%n]);
		}
	}
	struct cmp{
		Point p;
		cmp(const Point &p0){p = p0;}
		bool operator()(const Point &aa,const Point &bb){
			Point a = aa, b = bb;
			int d = sgn((a-p)^(b-p));
			if(d == 0){
				return sgn(a.distance(p)-b.distance(p)) < 0;
			}
			return d > 0;
		}
	};
	//`进行极角排序`
	//`首先需要找到最左下角的点`
	//`需要重载号好Point的 < 操作符(min函数要用) `
	void norm(){
		Point mi = p[0];
		for(int i = 1;i < n;i++)mi = min(mi,p[i]);
		sort(p,p+n,cmp(mi));
	}
	//`得到凸包`
	//`得到的凸包里面的点编号是0$\sim$n-1的`
	//`两种凸包的方法`
	//`注意如果有影响，要特判下所有点共点，或者共线的特殊情况`
	//`测试 LightOJ1203  LightOJ1239`
	void getconvex(polygon &convex){
		sort(p,p+n);
		convex.n = n;
		for(int i = 0;i < min(n,2);i++){
			convex.p[i] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//特判
		if(n <= 2)return;
		int &top = convex.n;
		top = 1;
		for(int i = 2;i < n;i++){
			while(top && sgn((convex.p[top]-p[i])^(convex.p[top-1]-p[i])) <= 0)
				top--;
			convex.p[++top] = p[i];
		}
		int temp = top;
		convex.p[++top] = p[n-2];
		for(int i = n-3;i >= 0;i--){
			while(top != temp && sgn((convex.p[top]-p[i])^(convex.p[top-1]-p[i])) <= 0)
				top--;
			convex.p[++top] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//特判
		convex.norm();//`原来得到的是顺时针的点，排序后逆时针`
	}
	//`得到凸包的另外一种方法`
	//`测试 LightOJ1203  LightOJ1239`
	void Graham(polygon &convex){
		norm();
		int &top = convex.n;
		top = 0;
		if(n == 1){
			top = 1;
			convex.p[0] = p[0];
			return;
		}
		if(n == 2){
			top = 2;
			convex.p[0] = p[0];
			convex.p[1] = p[1];
			if(convex.p[0] == convex.p[1])top--;
			return;
		}
		convex.p[0] = p[0];
		convex.p[1] = p[1];
		top = 2;
		for(int i = 2;i < n;i++){
			while( top > 1 && sgn((convex.p[top-1]-convex.p[top-2])^(p[i]-convex.p[top-2])) <= 0 )
				top--;
			convex.p[top++] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//特判
	}
	//`判断是不是凸的`
	bool isconvex(){
		bool s[2];
		memset(s,false,sizeof(s));
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			int k = (j+1)%n;
			s[sgn((p[j]-p[i])^(p[k]-p[i]))+1] = true;
			if(s[0] && s[2])return false;
		}
		return true;
	}
	//`判断点和任意多边形的关系`
	//` 3 点上`
	//` 2 边上`
	//` 1 内部`
	//` 0 外部`
	int relationpoint(Point q){
		for(int i = 0;i < n;i++){
			if(p[i] == q)return 3;
		}
		getline();
		for(int i = 0;i < n;i++){
			if(l[i].pointonseg(q))return 2;
		}
		int cnt = 0;
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			int k = sgn((q-p[j])^(p[i]-p[j]));
			int u = sgn(p[i].y-q.y);
			int v = sgn(p[j].y-q.y);
			if(k > 0 && u < 0 && v >= 0)cnt++;
			if(k < 0 && v < 0 && u >= 0)cnt--;
		}
		return cnt != 0;
	}
	//`直线u切割凸多边形左侧`
	//`注意直线方向`
	//`测试：HDU3982`
	void convexcut(Line u,polygon &po){
		int &top = po.n;//注意引用
		top = 0;
		for(int i = 0;i < n;i++){
			int d1 = sgn((u.e-u.s)^(p[i]-u.s));
			int d2 = sgn((u.e-u.s)^(p[(i+1)%n]-u.s));
			if(d1 >= 0)po.p[top++] = p[i];
			if(d1*d2 < 0)po.p[top++] = u.crosspoint(Line(p[i],p[(i+1)%n]));
		}
	}
	//`得到周长`
	//`测试 LightOJ1239`
	double getcircumference(){
		double sum = 0;
		for(int i = 0;i < n;i++){
			sum += p[i].distance(p[(i+1)%n]);
		}
		return sum;
	}
	//`得到面积`
	double getarea(){
		double sum = 0;
		for(int i = 0;i < n;i++){
			sum += (p[i]^p[(i+1)%n]);
		}
		return fabs(sum)/2;
	}
	//`得到方向`
	//` 1 表示逆时针，0表示顺时针`
	bool getdir(){
		double sum = 0;
		for(int i = 0;i < n;i++)
			sum += (p[i]^p[(i+1)%n]);
		if(sgn(sum) > 0)return 1;
		return 0;
	}
	//`得到重心`
	Point getbarycentre(){
		Point ret(0,0);
		double area = 0;
		for(int i = 1;i < n-1;i++){
			double tmp = (p[i]-p[0])^(p[i+1]-p[0]);
			if(sgn(tmp) == 0)continue;
			area += tmp;
			ret.x += (p[0].x+p[i].x+p[i+1].x)/3*tmp;
			ret.y += (p[0].y+p[i].y+p[i+1].y)/3*tmp;
		}
		if(sgn(area)) ret = ret/area;
		return ret;
	}
	//`多边形和圆交的面积`
	//`测试：POJ3675 HDU3982 HDU2892`
	double areacircle(circle c){
		double ans = 0;
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			if(sgn( (p[j]-c.p)^(p[i]-c.p) ) >= 0)
				ans += c.areatriangle(p[i],p[j]);
			else ans -= c.areatriangle(p[i],p[j]);
		}
		return fabs(ans);
	}
	//`多边形和圆关系`
	//` 2 圆完全在多边形内`
	//` 1 圆在多边形里面，碰到了多边形边界`
	//` 0 其它`
	int relationcircle(circle c){
		getline();
		int x = 2;
		if(relationpoint(c.p) != 1)return 0;//圆心不在内部
		for(int i = 0;i < n;i++){
			if(c.relationseg(l[i])==2)return 0;
			if(c.relationseg(l[i])==1)x = 1;
		}
		return x;
	}
};
//`AB X AC`
double cross(Point A,Point B,Point C){
	return (B-A)^(C-A);
}
//`AB*AC`
double dot(Point A,Point B,Point C){
	return (B-A)*(C-A);
}
//`最小矩形面积覆盖`
//` A 必须是凸包(而且是逆时针顺序)`
//` 测试 UVA 10173`
double minRectangleCover(polygon A){
	//`要特判A.n < 3的情况`
	if(A.n < 3)return 0.0;
	A.p[A.n] = A.p[0];
	double ans = -1;
	int r = 1, p = 1, q;
	for(int i = 0;i < A.n;i++){
		//`卡出离边A.p[i] - A.p[i+1]最远的点`
		while( sgn( cross(A.p[i],A.p[i+1],A.p[r+1]) - cross(A.p[i],A.p[i+1],A.p[r]) ) >= 0 )
			r = (r+1)%A.n;
		//`卡出A.p[i] - A.p[i+1]方向上正向n最远的点`
		while(sgn( dot(A.p[i],A.p[i+1],A.p[p+1]) - dot(A.p[i],A.p[i+1],A.p[p]) ) >= 0 )
			p = (p+1)%A.n;
		if(i == 0)q = p;
		//`卡出A.p[i] - A.p[i+1]方向上负向最远的点`
		while(sgn(dot(A.p[i],A.p[i+1],A.p[q+1]) - dot(A.p[i],A.p[i+1],A.p[q])) <= 0)
			q = (q+1)%A.n;
		double d = (A.p[i] - A.p[i+1]).len2();
		double tmp = cross(A.p[i],A.p[i+1],A.p[r]) *
			(dot(A.p[i],A.p[i+1],A.p[p]) - dot(A.p[i],A.p[i+1],A.p[q]))/d;
		if(ans < 0 || ans > tmp)ans = tmp;
	}
	return ans;
}

//`直线切凸多边形`
//`多边形是逆时针的，在q1q2的左侧`
//`测试:HDU3982`
vector<Point> convexCut(const vector<Point> &ps,Point q1,Point q2){
	vector<Point>qs;
	int n = ps.size();
	for(int i = 0;i < n;i++){
		Point p1 = ps[i], p2 = ps[(i+1)%n];
		int d1 = sgn((q2-q1)^(p1-q1)), d2 = sgn((q2-q1)^(p2-q1));
		if(d1 >= 0)
			qs.push_back(p1);
		if(d1 * d2 < 0)
			qs.push_back(Line(p1,p2).crosspoint(Line(q1,q2)));
	}
	return qs;
}
//`半平面交`
//`测试 POJ3335 POJ1474 POJ1279`
//***************************
struct halfplane:public Line{
	double angle;
	halfplane(){}
	//`表示向量s->e逆时针(左侧)的半平面`
	halfplane(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	halfplane(Line v){
		s = v.s;
		e = v.e;
	}
	void calcangle(){
		angle = atan2(e.y-s.y,e.x-s.x);
	}
	bool operator <(const halfplane &b)const{
		return angle < b.angle;
	}
};
struct halfplanes{
	int n;
	halfplane hp[maxp];
	Point p[maxp];
	int que[maxp];
	int st,ed;
	void push(halfplane tmp){
		hp[n++] = tmp;
	}
	//去重
	void unique(){
		int m = 1;
		for(int i = 1;i < n;i++){
			if(sgn(hp[i].angle-hp[i-1].angle) != 0)
				hp[m++] = hp[i];
			else if(sgn( (hp[m-1].e-hp[m-1].s)^(hp[i].s-hp[m-1].s) ) > 0)
				hp[m-1] = hp[i];
		}
		n = m;
	}
	bool halfplaneinsert(){
		for(int i = 0;i < n;i++)hp[i].calcangle();
		sort(hp,hp+n);
		unique();
		que[st=0] = 0;
		que[ed=1] = 1;
		p[1] = hp[0].crosspoint(hp[1]);
		for(int i = 2;i < n;i++){
			while(st<ed && sgn((hp[i].e-hp[i].s)^(p[ed]-hp[i].s))<0)ed--;
			while(st<ed && sgn((hp[i].e-hp[i].s)^(p[st+1]-hp[i].s))<0)st++;
			que[++ed] = i;
			if(hp[i].parallel(hp[que[ed-1]]))return false;
			p[ed]=hp[i].crosspoint(hp[que[ed-1]]);
		}
		while(st<ed && sgn((hp[que[st]].e-hp[que[st]].s)^(p[ed]-hp[que[st]].s))<0)ed--;
		while(st<ed && sgn((hp[que[ed]].e-hp[que[ed]].s)^(p[st+1]-hp[que[ed]].s))<0)st++;
		if(st+1>=ed)return false;
		return true;
	}
	//`得到最后半平面交得到的凸多边形`
	//`需要先调用halfplaneinsert() 且返回true`
	void getconvex(polygon &con){
		p[st] = hp[que[st]].crosspoint(hp[que[ed]]);
		con.n = ed-st+1;
		for(int j = st,i = 0;j <= ed;i++,j++)
			con.p[i] = p[j];
	}
};
//***************************

const int maxn = 1010;
struct circles{
	circle c[maxn];
	double ans[maxn];//`ans[i]表示被覆盖了i次的面积`
	double pre[maxn];
	int n;
	circles(){}
	void add(circle cc){
		c[n++] = cc;
	}
	//`x包含在y中`
	bool inner(circle x,circle y){
		if(x.relationcircle(y) != 1)return 0;
		return sgn(x.r-y.r)<=0?1:0;
	}
	//圆的面积并去掉内含的圆
	void init_or(){
		bool mark[maxn] = {0};
		int i,j,k=0;
		for(i = 0;i < n;i++){
			for(j = 0;j < n;j++)
				if(i != j && !mark[j]){
					if( (c[i]==c[j])||inner(c[i],c[j]) )break;
				}
			if(j < n)mark[i] = 1;
		}
		for(i = 0;i < n;i++)
			if(!mark[i])
				c[k++] = c[i];
		n = k;
	}
	//`圆的面积交去掉内含的圆`
	void init_add(){
		int i,j,k;
		bool mark[maxn] = {0};
		for(i = 0;i < n;i++){
			for(j = 0;j < n;j++)
				if(i != j && !mark[j]){
					if( (c[i]==c[j])||inner(c[j],c[i]) )break;
				}
			if(j < n)mark[i] = 1;
		}
		for(i = 0;i < n;i++)
			if(!mark[i])
				c[k++] = c[i];
		n = k;
	}
	//`半径为r的圆，弧度为th对应的弓形的面积`
	double areaarc(double th,double r){
		return 0.5*r*r*(th-sin(th));
	}
	//`测试SPOJVCIRCLES SPOJCIRUT`
	//`SPOJVCIRCLES求n个圆并的面积，需要加上init\_or()去掉重复圆（否则WA）`
	//`SPOJCIRUT 是求被覆盖k次的面积，不能加init\_or()`
	//`对于求覆盖多少次面积的问题，不能解决相同圆，而且不能init\_or()`
	//`求多圆面积并，需要init\_or,其中一个目的就是去掉相同圆`
	void getarea(){
		memset(ans,0,sizeof(ans));
		vector<pair<double,int> >v;
		for(int i = 0;i < n;i++){
			v.clear();
			v.push_back(make_pair(-pi,1));
			v.push_back(make_pair(pi,-1));
			for(int j = 0;j < n;j++)
				if(i != j){
					Point q = (c[j].p - c[i].p);
					double ab = q.len(),ac = c[i].r, bc = c[j].r;
					if(sgn(ab+ac-bc)<=0){
						v.push_back(make_pair(-pi,1));
						v.push_back(make_pair(pi,-1));
						continue;
					}
					if(sgn(ab+bc-ac)<=0)continue;
					if(sgn(ab-ac-bc)>0)continue;
					double th = atan2(q.y,q.x), fai = acos((ac*ac+ab*ab-bc*bc)/(2.0*ac*ab));
					double a0 = th-fai;
					if(sgn(a0+pi)<0)a0+=2*pi;
					double a1 = th+fai;
					if(sgn(a1-pi)>0)a1-=2*pi;
					if(sgn(a0-a1)>0){
						v.push_back(make_pair(a0,1));
						v.push_back(make_pair(pi,-1));
						v.push_back(make_pair(-pi,1));
						v.push_back(make_pair(a1,-1));
					}
					else{
						v.push_back(make_pair(a0,1));
						v.push_back(make_pair(a1,-1));
					}
				}
			sort(v.begin(),v.end());
			int cur = 0;
			for(int j = 0;j < v.size();j++){
				if(cur && sgn(v[j].first-pre[cur])){
					ans[cur] += areaarc(v[j].first-pre[cur],c[i].r);
					ans[cur] += 0.5*(Point(c[i].p.x+c[i].r*cos(pre[cur]),c[i].p.y+c[i].r*sin(pre[cur]))^Point(c[i].p.x+c[i].r*cos(v[j].first),c[i].p.y+c[i].r*sin(v[j].first)));
				}
				cur += v[j].second;
				pre[cur] = v[j].first;
			}
		}
		for(int i = 1;i < n;i++)
			ans[i] -= ans[i+1];
	}
};
```



## 三维几何

> 来自kuangbin

```cpp
const double eps = 1e-8;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0,double _y = 0,double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
	void output(){
		scanf("%.2lf %.2lf %.2lf\n",x,y,z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	bool operator <(const Point3 &b)const{
		return sgn(x-b.x)==0?(sgn(y-b.y)==0?sgn(z-b.z)<0:y<b.y):x<b.x;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//点乘
	double operator *(const Point3 &b)const{
		return x*b.x+y*b.y+z*b.z;
	}
	//叉乘
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
	double rad(Point3 a,Point3 b){
		Point3 p = (*this);
		return acos( ( (a-p)*(b-p) )/ (a.distance(p)*b.distance(p)) );
	}
	//变换长度
	Point3 trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point3(x*r,y*r,z*r);
	}
};
struct Line3
{
	Point3 s,e;
	Line3(){}
	Line3(Point3 _s,Point3 _e)
	{
		s = _s;
		e = _e;
	}
	bool operator ==(const Line3 v)
	{
		return (s==v.s)&&(e==v.e);
	}
	void input()
	{
		s.input();
		e.input();
	}
	double length()
	{
		return s.distance(e);
	}
	//点到直线距离
	double dispointtoline(Point3 p)
	{
		return ((e-s)^(p-s)).len()/s.distance(e);
	}
	//点到线段距离
	double dispointtoseg(Point3 p)
	{
		if(sgn((p-s)*(e-s)) < 0 || sgn((p-e)*(s-e)) < 0)
			return min(p.distance(s),e.distance(p));
		return dispointtoline(p);
	}
	//`返回点p在直线上的投影`
	Point3 lineprog(Point3 p)
	{
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`p绕此向量逆时针arg角度`
	Point3 rotate(Point3 p,double ang)
	{
		if(sgn(((s-p)^(e-p)).len()) == 0)return p;
		Point3 f1 = (e-s)^(p-s);
		Point3 f2 = (e-s)^(f1);
		double len = ((s-p)^(e-p)).len()/s.distance(e);
		f1 = f1.trunc(len); f2 = f2.trunc(len);
		Point3 h = p+f2;
		Point3 pp = h+f1;
		return h + ((p-h)*cos(ang)) + ((pp-h)*sin(ang));
	}
	//`点在直线上`
	bool pointonseg(Point3 p)
	{
		return sgn( ((s-p)^(e-p)).len() ) == 0 && sgn((s-p)*(e-p)) == 0;
	}
};
struct Plane
{
	Point3 a,b,c,o;//`平面上的三个点，以及法向量`
	Plane(){}
	Plane(Point3 _a,Point3 _b,Point3 _c)
	{
		a = _a;
		b = _b;
		c = _c;
		o = pvec();
	}
	Point3 pvec()
	{
		return (b-a)^(c-a);
	}
	//`ax+by+cz+d = 0`
	Plane(double _a,double _b,double _c,double _d)
	{
		o = Point3(_a,_b,_c);
		if(sgn(_a) != 0)
			a = Point3((-_d-_c-_b)/_a,1,1);
		else if(sgn(_b) != 0)
			a = Point3(1,(-_d-_c-_a)/_b,1);
		else if(sgn(_c) != 0)
			a = Point3(1,1,(-_d-_a-_b)/_c);
	}
	//`点在平面上的判断`
	bool pointonplane(Point3 p)
	{
		return sgn((p-a)*o) == 0;
	}
	//`两平面夹角`
	double angleplane(Plane f)
	{
		return acos(o*f.o)/(o.len()*f.o.len());
	}
	//`平面和直线的交点，返回值是交点个数`
	int crossline(Line3 u,Point3 &p)
	{
		double x = o*(u.e-a);
		double y = o*(u.s-a);
		double d = x-y;
		if(sgn(d) == 0)return 0;
		p = ((u.s*x)-(u.e*y))/d;
		return 1;
	}
	//`点到平面最近点(也就是投影)`
	Point3 pointtoplane(Point3 p)
	{
		Line3 u = Line3(p,p+o);
		crossline(u,p);
		return p;
	}
	//`平面和平面的交线`
	int crossplane(Plane f,Line3 &u)
	{
		Point3 oo = o^f.o;
		Point3 v = o^oo;
		double d = fabs(f.o*v);
		if(sgn(d) == 0)return 0;
		Point3 q = a + (v*(f.o*(f.a-a))/d);
		u = Line3(q,q+oo);
		return 1;
	}
};
```



## 平面最近点对

> 来自kuangbin

```cpp
const int MAXN = 100010;
const double eps = 1e-8;
const double INF = 1e20;
struct Point{
	double x,y;
	void input(){
		scanf("%lf%lf",&x,&y);
	}
};
double dist(Point a,Point b){
	return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}
Point p[MAXN];
Point tmpt[MAXN];
bool cmpx(Point a,Point b){
	return a.x < b.x || (a.x == b.x && a.y < b.y);
}
bool cmpy(Point a,Point b){
	return a.y < b.y || (a.y == b.y && a.x < b.x);
}
double Closest_Pair(int left,int right){
	double d = INF;
	if(left == right)return d;
	if(left+1 == right)return dist(p[left],p[right]);
	int mid = (left+right)/2;
	double d1 = Closest_Pair(left,mid);
	double d2 = Closest_Pair(mid+1,right);
	d = min(d1,d2);
	int cnt = 0;
	for(int i = left;i <= right;i++){
		if(fabs(p[mid].x - p[i].x) <= d)
			tmpt[cnt++] = p[i];
	}
	sort(tmpt,tmpt+cnt,cmpy);
	for(int i = 0;i < cnt;i++){
		for(int j = i+1;j < cnt && tmpt[j].y - tmpt[i].y < d;j++)
			d = min(d,dist(tmpt[i],tmpt[j]));
	}
	return d;
}
int main(){
	int n;
	while(scanf("%d",&n) == 1 && n){
		for(int i = 0;i < n;i++)p[i].input();
		sort(p,p+n,cmpx);
		printf("%.2lf\n",Closest_Pair(0,n-1));
	}
    return 0;
}
```



## 三维凸包

> 来自kuangbin

```cpp
const double eps = 1e-8;
const int MAXN = 550;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0, double _y = 0, double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//点乘
	double operator *(const Point3 &b)const{
		return x*b.x + y*b.y + z*b.z;
	}
	//叉乘
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
};
struct CH3D{
	struct face{
		//表示凸包一个面上的三个点的编号
		int a,b,c;
		//表示该面是否属于最终的凸包上的面
		bool ok;
	};
	//初始顶点数
	int n;
	Point3 P[MAXN];
	//凸包表面的三角形数
	int num;
	//凸包表面的三角形
	face F[8*MAXN];
	int g[MAXN][MAXN];
	//叉乘
	Point3 cross(const Point3 &a,const Point3 &b,const Point3 &c){
		return (b-a)^(c-a);
	}
	//`三角形面积*2`
	double area(Point3 a,Point3 b,Point3 c){
		return ((b-a)^(c-a)).len();
	}
	//`四面体有向面积*6`
	double volume(Point3 a,Point3 b,Point3 c,Point3 d){
		return ((b-a)^(c-a))*(d-a);
	}
	//`正：点在面同向`
	double dblcmp(Point3 &p,face &f){
		Point3 p1 = P[f.b] - P[f.a];
		Point3 p2 = P[f.c] - P[f.a];
		Point3 p3 = p - P[f.a];
		return (p1^p2)*p3;
	}
	void deal(int p,int a,int b){
		int f = g[a][b];
		face add;
		if(F[f].ok){
			if(dblcmp(P[p],F[f]) > eps)
				dfs(p,f);
			else {
				add.a = b;
				add.b = a;
				add.c = p;
				add.ok = true;
				g[p][b] = g[a][p] = g[b][a] = num;
				F[num++] = add;
			}
		}
	}
	//递归搜索所有应该从凸包内删除的面
	void dfs(int p,int now){
		F[now].ok = false;
		deal(p,F[now].b,F[now].a);
		deal(p,F[now].c,F[now].b);
		deal(p,F[now].a,F[now].c);
	}
	bool same(int s,int t){
		Point3 &a = P[F[s].a];
		Point3 &b = P[F[s].b];
		Point3 &c = P[F[s].c];
		return fabs(volume(a,b,c,P[F[t].a])) < eps &&
			fabs(volume(a,b,c,P[F[t].b])) < eps &&
			fabs(volume(a,b,c,P[F[t].c])) < eps;
	}
	//构建三维凸包
	void create(){
		num = 0;
		face add;

		//***********************************
		//此段是为了保证前四个点不共面
		bool flag = true;
		for(int i = 1;i < n;i++){
			if(!(P[0] == P[i])){
				swap(P[1],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		flag = true;
		for(int i = 2;i < n;i++){
			if( ((P[1]-P[0])^(P[i]-P[0])).len() > eps ){
				swap(P[2],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		flag = true;
		for(int i = 3;i < n;i++){
			if(fabs( ((P[1]-P[0])^(P[2]-P[0]))*(P[i]-P[0]) ) > eps){
				swap(P[3],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		//**********************************

		for(int i = 0;i < 4;i++){
			add.a = (i+1)%4;
			add.b = (i+2)%4;
			add.c = (i+3)%4;
			add.ok = true;
			if(dblcmp(P[i],add) > 0)swap(add.b,add.c);
			g[add.a][add.b] = g[add.b][add.c] = g[add.c][add.a] = num;
			F[num++] = add;
		}
		for(int i = 4;i < n;i++)
			for(int j = 0;j < num;j++)
				if(F[j].ok && dblcmp(P[i],F[j]) > eps){
					dfs(i,j);
					break;
				}
		int tmp = num;
		num = 0;
		for(int i = 0;i < tmp;i++)
			if(F[i].ok)
				F[num++] = F[i];
	}
	//表面积
	//`测试：HDU3528`
	double area(){
		double res = 0;
		if(n == 3){
			Point3 p = cross(P[0],P[1],P[2]);
			return p.len()/2;
		}
		for(int i = 0;i < num;i++)
			res += area(P[F[i].a],P[F[i].b],P[F[i].c]);
		return res/2.0;
	}
	double volume(){
		double res = 0;
		Point3 tmp = Point3(0,0,0);
		for(int i = 0;i < num;i++)
			res += volume(tmp,P[F[i].a],P[F[i].b],P[F[i].c]);
		return fabs(res/6);
	}
	//表面三角形个数
	int triangle(){
		return num;
	}
	//表面多边形个数
	//`测试：HDU3662`
	int polygon(){
		int res = 0;
		for(int i = 0;i < num;i++){
			bool flag = true;
			for(int j = 0;j < i;j++)
				if(same(i,j)){
					flag = 0;
					break;
				}
			res += flag;
		}
		return res;
	}
	//重心
	//`测试：HDU4273`
	Point3 barycenter(){
		Point3 ans = Point3(0,0,0);
		Point3 o = Point3(0,0,0);
		double all = 0;
		for(int i = 0;i < num;i++){
			double vol = volume(o,P[F[i].a],P[F[i].b],P[F[i].c]);
			ans = ans + (((o+P[F[i].a]+P[F[i].b]+P[F[i].c])/4.0)*vol);
			all += vol;
		}
		ans = ans/all;
		return ans;
	}
	//点到面的距离
	//`测试：HDU4273`
	double ptoface(Point3 p,int i){
		double tmp1 = fabs(volume(P[F[i].a],P[F[i].b],P[F[i].c],p));
		double tmp2 = ((P[F[i].b]-P[F[i].a])^(P[F[i].c]-P[F[i].a])).len();
		return tmp1/tmp2;
	}
};
CH3D hull;
int main()
{
    while(scanf("%d",&hull.n) == 1){
		for(int i = 0;i < hull.n;i++)hull.P[i].input();
		hull.create();
		Point3 p = hull.barycenter();
		double ans = 1e20;
		for(int i = 0;i < hull.num;i++)
			ans = min(ans,hull.ptoface(p,i));
		printf("%.3lf\n",ans);
	}
    return 0;
}
```



# 其他



## 快读



### 普通快读

```cpp
template <typename T>
inline void read(T &x) {
    x = 0; int f = 1; char ch = getchar();
    while (!isdigit(ch)) { if (ch == '-') f = -1; ch = getchar(); }
    while (isdigit(ch)) { x = (x << 3) + (x << 1) + (ch & 15); ch = getchar(); }
    x *= f;
}
```



### 文件加速

```cpp
namespace _{  
char buf[100000], *p1 = buf, *p2 = buf; bool rEOF = 1;//为0表示文件结尾  
    inline char nc() { return p1 == p2 && rEOF && (p2 = (p1 = buf) + fread(buf, 1, 100000, stdin), p1 == p2) ? (rEOF = 0, EOF) : *p1++; }  
    template<class _T>  
    inline void read(_T &num){  
       char c = nc(), f = 1; num = 0;  
       while (c<'0' || c>'9')c == '-' && (f = -1), c = nc();  
       while (c >= '0'&&c <= '9')num = num * 10 + c - '0', c = nc();  
       num *= f;  
   }  
   inline bool need(char &c){ return c >= 'a'&&c <= 'z' || c >= '0'&&c <= '9' || c >= 'A'&&c <= 'Z'; }//读入的字符范围  
   inline void read_str(char *a){  
       while ((*a = nc()) && !need(*a) && rEOF);   ++a;  
       while ((*a = nc()) && need(*a) && rEOF)++a; --p1, *a = '\0';  
   }  
}using namespace _;  
```

