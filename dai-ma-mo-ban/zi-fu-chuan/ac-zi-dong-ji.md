# AC自动机

```cpp
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 2e5 + 10;
const int M = 2e6 + 10;

int n;
int tr[N][26], tp[N], pt[N], ne[N], idx;
char str[M];
int q[N], siz[N];

void insert (int t) {
    int p = 0;
    for (int i = 0; str[i]; ++ i) {
        int u = str[i] - 'a';
        if (!tr[p][u]) tr[p][u] = ++ idx;
        p = tr[p][u];
    }
    tp[t] = p; //第t个插入的字符串的终止结点在p处 
    pt[p] = t; //结点p所对应的字符串是t【如果有的话】 
}

void build () {
    int hh = 0, tt = -1;
    for (int i = 0; i < 26; ++ i)
        if (tr[0][i]) q[++ tt] = tr[0][i]; //如果有以这个字母开头的字符串就将结点加入队列

    while (hh <= tt) {
        int u = q[hh ++]; //u是上一层的结点 
        for (int i = 0; i < 26; ++ i) { 
            int p = tr[u][i];
            if (!p) tr[u][i] = tr[ne[u]][i]; //如果没有这个字母，就虚构一个点指向上一层的这个字母 
            else {
                ne[p] = tr[ne[u]][i]; //失配位置就是上一层结点的失配位置的对应字母 
                q[++ tt] = p;  
            } 
        } 
    } 
}

void query () {
    int p = 0; //初始化为根节点 
    for (int i = 0; str[i]; ++ i) {
        int u = str[i] - 'a';
        p = tr[p][u]; //跳到Trie树对应的位置 

        ++ siz[p]; //这个结点匹配了一次 
    }
}

vector<int> G[N];

void dfs (int x) {
    for (int p : G[x]) {
        dfs (p);
        siz[x] += siz[p];
    }
} 

int main () {
    scanf ("%d", &n);
    for (int i = 1; i <= n; ++ i) {
        scanf ("%s", str);
        insert (i);
    }

    build ();

    scanf ("%s", str);
    query ();

    for (int i = 1; i <= idx; ++ i) 
        G[ne[i]].push_back(i); //fail树 

    dfs (0);

    for (int i = 1; i <= n; ++ i) 
        printf ("%d\n", siz[tp[i]]);

    return 0;
}
```

