# 树状数组

> 通过lowbit来跳转层数达到维护区间的目的

```cpp
inline int lowbit(int x) {
    return x & (-x);
}

void add(int x, int k) {
    for (; x <= n; x += lowbit(x))
        tr[x] += k;
}

int query(int x) {
    int ans = 0;
    for (; x; x -= lowbit(x))
        ans += tr[x];
       return ans;
}
```

