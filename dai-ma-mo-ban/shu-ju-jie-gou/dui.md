# 堆

## 二叉堆

```cpp
const int N = 1000000 + 10;

template <class T>
class binary_heap {
    private :
        vector<T> h; 
        int s = 0;
        inline void up (int u) {
            while (u > 1 && h[u] < h[u/2]) 
                swap (h[u] , h[u/2]), u >>= 1;
        }
        inline void down (int u) {
            while (u * 2 <= s) {
                int t = u * 2;
                if (t + 1 <= s && h[t + 1] < h[t]) t++;
                if (h[t] >= h[u]) break;
                swap(h[u], h[t]);
                u = t;
              }
        }
    public :
        binary_heap () { h.reserve (N); }
        binary_heap (T arr[] , int len) { 
            for (int i = 1 ; i <= len ; ++ i) h[i] = arr[i];
            for (int i = s / 2 ; i ; -- i) down (i); 
        }
        inline void push (T x) { h[++s] = x , up(s); }
        inline void pop () { swap (h[1] , h[s--]) , down (1); }
        inline T top () { return h[1]; }
};
```

## 模拟堆笔记

### 样例

#### Input

```text
8
I -10
PM
I -10
D 1
C 2 8
I 6
PM
DM
```

#### output

```text
-10
6
```

### 思路

> 很明显，需要记录第k个插入的数在堆的哪个位置，`ph[i]`表示第i个数在哪个位置
>
> 所以在交换的时候需要写额外的交换函数`heap_swap`
>
> 在写的过程中可知我们还需要知道下标为i的数是第几个插入的，`hp[i]`表示第i个数是第几个插入的
>
> ```cpp
> void heap_swap (int a, int b) {
>     swap(ph[hp[a]], ph[hp[b]]);
>     swap(hp[a], hp[b]);
>     swap(h[a], h[b]);
> }
> ```
>
> 然后就是套堆的维护板子
>
> ```cpp
> void up (int u) {
>     while (u / 2 && h[u] < h[u / 2]) {
>         heap_swap (u, u / 2);
>         u /= 2;
>     }
> }
>
> void down (int u) {
>     int t = u;
>     if (u * 2 <= s && h[u * 2] < h[t]) t = u * 2;
>     if (u * 2 + 1 <= s && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
>     if (t != u) {
>         heap_swap (u, t);
>         down (t);
>     }
> }
> ```
>
> 最后写上对命令判断执行对应操作即可
>
> 注意：在删除的时候，需要记录删除的点的原始的位置，即记录删除的第k个点所在的位置，否则swap的时候ph也会变化，后期无法维护整个堆

### 代码

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N = 1e5;

int n;
int h[N], ph[N], hp[N], s;
//ph是第k个插入的数的下标
//hp是堆的当前状态的插入次序

void heap_swap (int a, int b) {
    swap (ph[hp[a]], ph[hp[b]]);
    swap (hp[a], hp[b]);
    swap (h[a], h[b]);
}
void down (int u) {
    int t = u;
    if (u * 2 <= s && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= s && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (t != u) {
        heap_swap (t , u);
        down (t);
    }
}

void up (int u) {
    while (u / 2 && h[u] < h[u / 2]) {
        heap_swap (u, u / 2);
        u /= 2;
    }
}

int main () {
    int m = 0;
    scanf("%d", &n);
    while (n -- ) {
        char opt[5];
        scanf("%s", opt);
        if (opt[0] == 'I') {
            int x;
            m ++;
            scanf ("%d", &x);
            ph[m] = ++ s; hp[s] = m; h[s] = x;
            up (s);
        } else if (opt[0] == 'C') {
            int k, x;
            scanf("%d%d", &k, &x);
            h[ph[k]] = x;
            up (ph[k]), down (ph[k]);
        } else if (opt[0] == 'P') {
            printf ("%d\n", h[1]);
        } else if (opt[1] == 'M') {
            heap_swap (1, s); s--;
            down (1);
        } else {
            int k;
            scanf("%d", &k);
            int u = ph[k];
            heap_swap (u, s); s --;
            up(u); down(u);
        }
    }
}
```

