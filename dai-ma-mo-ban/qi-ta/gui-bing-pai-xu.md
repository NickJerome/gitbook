# 归并排序

## 归并排序

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e6 + 10;

int n;
int a[N], res[N];

void merge_sort(int q[], int l, int r) {
    if (l >= r) return ;
    int mid = l + r >> 1;
    merge_sort(q, l, mid); merge_sort(q, mid+1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] < q[j]) res[++ k] = q[i ++];
        else res[++ k] = q[j ++];
    while (i <= mid) res[++ k] = q[i ++];
    while (j <= r) res[++ k] = q[j ++];

    for (i = l, j = 1; i <= r; ++ i, ++ j) q[i] = res[j];
}

int main () {
    scanf("%d", &n);
    for (int i = 1; i <= n; ++ i) scanf("%d", &a[i]);
    merge_sort(a, 1, n);
    for (int i = 1; i <= n; ++ i) printf("%d ", a[i]);
    return 0;
}
```

## 归并排序求逆序对

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
    LL ans = merge_sort(q, l, mid) + merge_sort(q, mid+1, r);

    int k = 0, i = l, j = mid + 1;
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

