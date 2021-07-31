# 逆序对

> 可以用来求解冒泡排序的最小交换次数



## 归并排序求逆序数

```cpp
#include <iostream>
#include <algorithm>
using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n;
int a[N], res[N];

LL merge_sort(int q[], int l, int r) {
    if (l >= r) return 0;
    
    int mid = l + r >> 1;
    LL ans = merge_sort(q, l, mid) + merge_sort(q, mid + 1, r);
    
    int i = l, j = mid + 1, k = 0;
    while (i <= mid && j <= r) {
        if (q[i] <= q[j]) res[++ k] = q[i ++];
        else {
            ans += mid - i + 1;
            res[++ k] = q[j ++];
        }
    }
    while (i <= mid) res[++ k] = q[i ++];
    while (j <= r) res[++ k] = q[j ++];
    
    for (i = l, j = 1; i <= r; ++ i, ++ j) q[i] = res[j];
    
    return ans;
}

int main () {
    scanf ("%d", &n);
    for (int i = 1; i <= n; ++ i) scanf ("%d", &a[i]);
    printf ("%lld" , merge_sort (a, 1, n));
    return 0;
}
```



## 树状数组求逆序数

```cpp
#include <iostream>
#include <algorithm>
using namespace std;

typedef long long LL;

const int N = 1e5 + 10;

int n;
int a[N];
vector<int> ve;
unordered_map<int, int> mp;

struct Segment_Tree {
    int tr[N];
    void add(int x, int v) {
        for (; x < N; x += x & (-x)) tr[x] += v;
    }
    int ask(int x) {
        int res = 0;
        for (; x; x -= x & (-x)) res += tr[x];
        return res;
    }
} seg;

int main () {
    scanf ("%d", &n);
    for (int i = 1; i <= n; ++ i) {
        scanf ("%d", &a[i]);
        ve.push_back(a[i]);
    }
    
    //离散化点
    sort(ve.begin(), ve.end());
    ve.erase(unique(ve.begin(), ve.end()), ve.end());
    int len = ve.size();
    for (int i = 0; i < len; ++ i) mp[ve[i]] = i + 1;
    
    LL res = 0;
    for (int i = 1; i <= n; ++ i) {
        int x = a[i];
        int p = mp[x];
        seg.add(p, 1);
        res += i - seg.ask(p);
    }
    
    printf("%lld\n", res);   
    
    return 0;
}
```

