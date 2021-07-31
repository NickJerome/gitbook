# 单调队列

> 滑动窗口的实现

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

