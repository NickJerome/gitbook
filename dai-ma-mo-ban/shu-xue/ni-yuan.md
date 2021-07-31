# 逆元

> 即$AX=1$，即$X=A^{-1}$，求$X$ 一般是在$\bmod p$的意义下的逆元，$AX=1 \(\bmod p\)$，求$A^{-1}$

## 费马小定理

> $\because ax = 1 \(\bmod p\)$ $ \therefore ax=a^{p-1} \(\bmod p\)$ $\therefore x = a^{p-2} \(\bmod p\)$
>
> $p$ 得是一个素数才可以用

```cpp
typedef long long LL;

LL qmi (LL a , LL b , LL p) {
    LL res = 1 % p;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return res % p;
}

LL inv(LL n, LL p) {
    return qmi(n, p - 2, p);
}
```

## 扩展欧几里德

> $\because ax = 1 \(\bmod p\)$ $\therefore ax = bp + 1$ $\therefore ax-bp = 1$

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
int inv(int n, int p) {
    int x, y;
       int g = exgcd(n, p, x, y);
    if(g == 1) return (x % p + p) % p;
    else return -1;
}
```

## 简单写法

> 只能求$a &lt; m$的情况，且必须保证 $a$ 与 $m$ 互质

```cpp
typedef long long LL;
LL inv(LL a, LL p) {
    if (a == 1) return 1;
    return inv(p % a, p) * (p - p / a) % p;
}
```

