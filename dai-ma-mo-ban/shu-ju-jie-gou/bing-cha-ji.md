# 并查集

## 常规写法

```cpp
int fa[N];

void init () {
    for (int i = 1; i <= n; ++ i) fa[i] = i;
}
int find(int x) {
    if (fa[x] == x) return x;
    return fa[x] = find(fa[x]);
}
```

## 一句话写法

```cpp
int p[N];

void init () {
    for (int i = 1; i <= n; ++ i) p[i] = i;
}
int find(int x) {
    return p[x] == x ? x : p[x] = find(p[x]);
}
```

## 带权并查集

> 维护两个点之间的信息，可以是距离，奇偶状态等等

```cpp
int p[N];
void init() {
    for (int i = 1; i <= n; ++ i) p[i] = i;
}
int find(int x){
    if (p[x] == x) return x;
    else {
        int root = find(p[x]);
        // Todo 如：d[x] += d[p[x]];
        return p[x] = root;
    }
}
```

## 并查集例题
https://acm.hdu.edu.cn/showproblem.php?pid=7001
> 可能卡常,用fread读入优化
> 由于查询区间为1到i-1, 所以可以考虑用并查集,在位置x修改为1的时候与后一位合并
> 查询的时候查询1是否存在,如果结果等于x的时候,说明如果此时x位置改成1,i的位置还能往后移动,所以i更新为find(x+1)

```cpp
#include <bits/stdc++.h>
using namespace std;

namespace _{  
    char buf[100000], *p1 = buf, *p2 = buf; bool rEOF = 1;//为0表示文件结尾  
    inline char nc(){ return p1 == p2 && rEOF && (p2 = (p1 = buf) + fread(buf, 1, 100000, stdin), p1 == p2) ? (rEOF = 0, EOF) : *p1++; }  
    template<class _T>  
    inline void read(_T &num){  
        char c = nc(), f = 1; num = 0;  
        while (c<'0' || c>'9')c == '-' && (f = -1), c = nc();  
        while (c >= '0'&&c <= '9')num = num * 10 + c - '0', c = nc();  
        num *= f;  
    }  
    inline bool need(char &c){ return c >= 'a'&&c <= 'z' || c >= '0'&&c <= '9' || c >= 'A'&&c <= 'Z'; }//读入的字符范围  
    inline void read_str(char *a){  
        while ((*a = nc()) && !need(*a) && rEOF);   ++a;  
        while ((*a = nc()) && need(*a) && rEOF)++a; --p1, *a = '\0';  
    }  
}using namespace _;


const int N = 5e6 + 10;

int n;
int p[N];

int find(int x) {
    if (p[x] == x) return x;
    return p[x] = find(p[x]);
}

int main () {
    int n;
    read(n);
    
    for (int i = 1; i <= n; ++ i) p[i] = i;
    
    while (n --) {
        int opt, x;
        read(opt); read(x);
        if (opt == 1) {
            p[find(x)] = find(x + 1);
        } else {
            int i = find(1);
            if (i == x) i = find(x + 1);
            printf ("%d\n", i);
        }
    }
    return 0;
}
```
