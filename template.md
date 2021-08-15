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




# 图论



# 数学



# 计算几何



# 其他