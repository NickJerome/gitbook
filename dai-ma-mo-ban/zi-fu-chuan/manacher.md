# Manacher

> 一种用于找回文串的线性算法

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 2e7 + 10;

int n;
char a[N], b[N];
int p[N];

void init() {
    int k = 0;
    b[k ++] = '$'; b[k ++] = '#';
    for (int i = 0; a[i]; ++ i) b[k ++] = a[i], b[k ++] = '#';
    b[k ++] = '^';
    n = k;
}

void manacher() {
    int mr = 0, mid;
    for (int i = 1; i < n; ++ i) {
        if (i < mr) p[i] = min(p[mid * 2 - i], mr - i);
        else p[i] = 1;
        while (b[i - p[i]] == b[i + p[i]]) p[i] ++;
        if (i + p[i] > mr) {
             mr = i + p[i];
             mid = i;
        }
    }
}

int main() {
    scanf("%s", a);

    init();
    manacher();

    int res = 0;
    for (int i = 0; i < n; i ++ ) 
        res = max(res, p[i]);

    printf("%d\n", res - 1);

    return 0;
}
```

