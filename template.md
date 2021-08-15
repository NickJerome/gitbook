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

