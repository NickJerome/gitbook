# 字典树

> Trie树，通过将每个字母当作一个节点建一棵树，在终止位置打标记即可

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N = 2e4 + 10;
const int M = 26 + 10;

int n;
char str[N];
int son[N][M], idx;
int cnt[N];

void insert (char str[]) {
    int p = 0;
    for (int i = 0; str[i]; ++ i) {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++;

}

int query (char str[]) {
    int p = 0;
    for (int i = 0; str[i]; ++ i) {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }

    return cnt[p];
}

int main () {
    scanf ("%d", &n);
    for (int i = 1; i <= n; ++ i) {
        char opt;
        scanf(" %c%s", &opt, str);
        if (opt == 'I') {
            insert (str);
        } else {
            printf ("%d\n", query (str));
        }
    }

    return 0;
}
```

