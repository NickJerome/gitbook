# 二叉搜索树

## Treap

### 旋转Treap

```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100000 + 10, INF = 1e8;

int n;
struct Node {
    int l, r;
    int key, val;
    int cnt, size;
} tr[N];

int root, idx;

void pushup(int p) {
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key) {
    tr[ ++ idx].key = key;
    tr[idx].val = rand();
    tr[idx].cnt = tr[idx].size = 1;
    return idx;
}

void zig(int &p) { // 右旋
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

void zag(int &p) { // 左旋
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build() {
    get_node(-INF), get_node(INF); //添加哨兵用于判断上下界
    root = 1, tr[1].r = 2;
    pushup(root);

    if (tr[1].val < tr[2].val) zag(root);
}


void insert(int &p, int key) {
    if (!p) p = get_node(key);
    else if (tr[p].key == key) tr[p].cnt ++ ;
    else if (tr[p].key > key) {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    } else {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
    pushup(p);
}

void remove(int &p, int key) {
    if (!p) return;
    if (tr[p].key == key) {
        if (tr[p].cnt > 1) tr[p].cnt -- ;
        else if (tr[p].l || tr[p].r) {
            if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val) {
                zig(p);
                remove(tr[p].r, key);
            } else {
                zag(p);
                remove(tr[p].l, key);
            }
        } else p = 0;
    } else if (tr[p].key > key) remove(tr[p].l, key);
    else remove(tr[p].r, key);

    pushup(p);
}

int get_rank_by_key(int p, int key) { // 通过数值找排名
    if (!p) return 0;   // 本题中不会发生此情况
    if (tr[p].key == key) return tr[tr[p].l].size + 1;
    if (tr[p].key > key) return get_rank_by_key(tr[p].l, key);
    return tr[tr[p].l].size + tr[p].cnt + get_rank_by_key(tr[p].r, key);
}

int get_key_by_rank(int p, int rank) { // 通过排名找数值
    if (!p) return INF;     // 本题中不会发生此情况
    if (tr[tr[p].l].size >= rank) return get_key_by_rank(tr[p].l, rank);
    if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
    return get_key_by_rank(tr[p].r, rank - tr[tr[p].l].size - tr[p].cnt);
}

int get_prev(int p, int key) { // 找到严格小于key的最大数
    if (!p) return -INF;
    if (tr[p].key >= key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key) { // 找到严格大于key的最小数
    if (!p) return INF;
    if (tr[p].key <= key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}

int main() {
    build(); //一定要build

    scanf("%d", &n);
    while (n -- ) {
        int opt, x;
        scanf("%d%d", &opt, &x);
        if (opt == 1) insert(root, x);
        else if (opt == 2) remove(root, x);
        else if (opt == 3) printf("%d\n", get_rank_by_key(root, x) - 1);
        else if (opt == 4) printf("%d\n", get_key_by_rank(root, x + 1));
        else if (opt == 5) printf("%d\n", get_prev(root, x));
        else printf("%d\n", get_next(root, x));
    }

    return 0;
}
```

### 无旋Treap \[FHQ Treap\]

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

const int N = 32767 + 10;
const int INF = 1e8;

struct Node {
    int l, r;
    int key, val;
    int size;
} tr[N];
int root, idx;

void pushup (int p) {
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + 1;
} 
int New (int key) {
    tr[++ idx].key = key;
    tr[idx].val = rand();
    tr[idx].size = 1;
    return idx;
}
void split (int p, int key, int &x, int &y) {
    if (!p) {
        x = y = 0;
        return ;
    }
    if (key < tr[p].key) {
        y = p;
        split (tr[p].l, key, x, tr[p].l);
    } else {
        x = p;
        split (tr[p].r, key, tr[p].r, y);
    }
    pushup (p);
}
int merge (int x, int y) {
    if (!x || !y) return x + y;
    if (tr[x].val < tr[y].val) {
        tr[x].r = merge(tr[x].r, y);
        pushup (x);
        return x;
    } else {
        tr[y].l = merge (x, tr[y].l);
        pushup (y);
        return y;
    }
}

void insert (int key) {
    int x, y;
    split (root, key, x, y);
    root = merge(merge(x, New (key)), y);
}

void remove (int key) {
    int x, y, z;
    split (root, key - 1, x, y);
    split (y, key, y, z);
    y = merge (tr[y].l, tr[y].r);
    root = merge(x, merge(y, z));
}

int Kth(int x,int k) {
    while(1)
    {
        int ls = tr[x].l,rs = tr[x].r;
        if(k<=tr[ls].size) x = ls;
        else if(k>tr[ls].size+1)
            k-=tr[ls].size+1,x = rs;
        else return x;
    }
}

int get_prev(int val) {
    int r1, r2;
    split(root, val-1, r1, r2);
    int ans = tr[Kth(r1,tr[r1].size)].key;
    root = merge(r1,r2);
    return ans;
}
int get_next(int val) {
    int r1, r2;
    split(root,val,r1,r2);
    int ans = tr[Kth(r2,1)].key;
    root =merge(r1,r2);
    return ans;
}


int main () {
    int n, res = 0;
    scanf ("%d", &n);
    insert (-INF); insert (INF);
    for (int i = 1; i <= n; ++ i) {
        int x;
        scanf ("%d", &x);
        if (i == 1) res += x;
        else {
            int pre = get_prev(x + 1), nxt = get_next(x - 1);
            int delta = min (abs(x - pre), abs(nxt - x));
            res += delta;
        }
        insert (x);
    }
    printf ("%d\n", res);
    return 0;
}
```

## Splay

> 通过双旋将点转移到根节点

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

