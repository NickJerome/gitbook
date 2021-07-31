# ST表

> 码量小

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

