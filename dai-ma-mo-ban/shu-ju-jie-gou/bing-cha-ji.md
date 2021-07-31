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

