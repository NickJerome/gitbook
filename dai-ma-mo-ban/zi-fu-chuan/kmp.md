# KMP

> 用于求一个模式串在母串上的匹配位置以及次数

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100000 + 10, M = 1000000 + 10;

int n, m;
int ne[N];
char s[M], p[N];

int main () {
    scanf("%d", &n);
    scanf("%s", p + 1);
    scanf("%d", &m);
    scanf("%s", s + 1);

    //KMP预处理
    for (int i = 2, j = 0; i <= n ; ++ i) {
        while (j && p[i] != p[j + 1]) j = ne[j];
        if (p[i] == p[j + 1]) ++ j;
        ne[i] = j;
    }

    //KMP匹配过程
    for (int i = 1, j = 0; i <= m ; ++ i) {
        while (j && s[i] != p[j + 1]) j = ne[j];
        if (s[i] == p[j + 1]) ++ j;
        if (j == n) {
            printf ("%d " , i - n);
            j = ne[j];
        }
    }

    return 0;
}
```

