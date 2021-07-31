# 最大公约数

> GCD原理： $a \| d$, $b\|d$, 令$a=b\*q+r$,容易得到$r\|d$,而$r=a \bmod b$ 所以gcd\(a, b\) = gcd\(b, a % b\)

```cpp
int gcd (int a , int b) {
    return b ? gcd (b , a % b) : a;
}
```

