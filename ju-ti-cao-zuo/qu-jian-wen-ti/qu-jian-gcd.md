# 区间GCD

> 由于区间gcd具有可加性，所以可以用线段树，ST等数据结构维护
>
> 重要：gcd(a, b) = gcd(a, b - a); gcd(a, b) = gcd(a, -b);
>
> $a_1 \bmod m == a_2 \bmod m == ... == a_n \bmod m$ 其中存在的最大m值等价于他们的差分序列的gcd
> 例题：[Problem - D - Codeforces](https://codeforces.com/contest/1549/problem/D)



## ST表求区间GCD

```cpp
#include <bits/stdc++.h>

using i64 = long long;

constexpr int N = 2e5;

std::vector<int> logn(N);

void solve() {
	int n;
	std::cin >> n;
	
	std::vector<i64> a(n);
	for (int i = 0; i < n; i++) {
		std::cin >> a[i];
	}
	
	n--;
	std::vector<std::vector<i64>> f(n + 1, std::vector<i64>(21, 0));
	for (int i = 0; i < n; i++) {
		f[i][0] = a[i + 1] - a[i];
	}
	
	for (int j = 1; j < 21; j++) {
		for (int i = 0; i + (1 << j) < n; i++) {
			f[i][j] = std::gcd(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
		}
	}
	
	std::function<i64(int, int)> query = [&](int l, int r) {
		int k = logn[r - l + 1];
		return std::gcd(f[l][k], f[r - (1 << k) + 1][k]);
	};
	
	int ans = 0;
	for (int l = 0, r = 0; r < n; r++) {
		while (l <= r && query(l, r) == 1) l ++;
		ans = std::max(ans, r - l + 1);
	}
	
	ans++;
	std::cout << ans << "\n";
} 

int main() {
	std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int t;
    std::cin >> t;
    
    for (int i = 2; i < N; i ++) {
    	logn[i] = logn[i / 2] + 1;
	}
    
    while (t--) {
    	solve();
	} 
	
	return 0;
}
```



## 线段树求区间GCD

```cpp
#include <bits/stdc++.h>
using namespace std;
 
using i64 = long long;
 
constexpr int N = 3e5 + 10;
 
i64 w[N];
struct Node {
    int l, r;
    i64 sum, d;
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
            i64 b = w[r] - w[r - 1];
            tr[u] = {l, r, b, b};
        } else {
            tr[u] = {l, r};
            int mid = l + r >> 1;
            build(u << 1, l, mid); build(u << 1 | 1, mid + 1, r);
            pushup(u);
        }
    }
    void modify(int u, int x, i64 v) {
        if (tr[u].l == x && tr[u].r == x) {
            i64 b = tr[u].sum + v;
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

i64 query(int l, int r) {
	auto left = seg.query(1, 1, l);
	Node right({0, 0, 0, 0});
	if (l + 1 <= r) right = seg.query(1, l + 1, r);
	return abs(gcd(left.sum, right.d));
}
 
void solve() {
	int n;
	scanf("%d", &n);
    vector<i64> a(n + 1);
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



> 线段树维护的区间GCD支持区间修改和区间查询
>
> [246. 区间最大公约数 - AcWing题库](https://www.acwing.com/problem/content/description/247/)

```cpp
#include <bits/stdc++.h>
using namespace std;

using i64 = long long;

constexpr int N = 5e5 + 10;

i64 gcd(i64 a, i64 b) {
    return b ? gcd(b, a % b) : a;
}

i64 w[N];
struct Node {
    int l, r;
    i64 sum, d;
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
            i64 b = w[r] - w[r - 1];
            tr[u] = {l, r, b, b};
        } else {
            tr[u] = {l, r};
            int mid = l + r >> 1;
            build(u << 1, l, mid); build(u << 1 | 1, mid + 1, r);
            pushup(u);
        }
    }
    void modify(int u, int x, i64 v) {
        if (tr[u].l == x && tr[u].r == x) {
            i64 b = tr[u].sum + v;
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

void solve() {
    int n, m;
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) {
        scanf("%lld", &w[i]);
    }
    seg.build(1, 1, n);
    
    while (m --) {
        char op; int l, r;
        scanf(" %c%d%d", &op, &l, &r);
        if (op == 'C') {
            i64 x;
            scanf("%lld", &x);
            seg.modify(1, l, x);
            if (r + 1 <= n) seg.modify(1, r + 1, -x);
        } else {
            auto left = seg.query(1, 1, l);
            Node right({0, 0, 0, 0});
            if (l + 1 <= r) right = seg.query(1, l + 1, r);
            printf("%lld\n", abs(gcd(left.sum, right.d)));
        }
    }
}

int main() {
    solve();
    return 0;
}
```

