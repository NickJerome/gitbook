# 扩展欧几里得

> 同最大公约数类似，替换a,b求解

```cpp
int exgcd (int a, int b, int &x, int &y) {
    if (!b) {
        x = 1; y = 0;
        return a;
    }
    int g = exgcd (b, a % b, y, x);
    y -= a / b * x;
    return g;
}
```

