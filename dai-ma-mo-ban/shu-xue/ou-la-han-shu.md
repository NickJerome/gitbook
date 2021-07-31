# 欧拉函数

> 即求一个数n，在$1 \to n$中与n互质的数的个数

## 求值

```cpp
int phi (int x) {
    int res = x;
    for (int i = 2; i <= x / i; ++ i) 
        if (x % i == 0) {
            res = res / i * (i - 1);
            while (x % i == 0) x /= i;
        }
    if (x > 1) res = res / x * (x - 1);

    return res;
}
```

## 筛法

```cpp
void get_eulers (int n) {
    phi[1] = 1;
    for (int i = 2; i <= n; ++ i) {
        if (!vis[i]) {
            primes[cnt ++] = i;
            phi[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++) {
            vis[i * primes[j]] = true;
            if (i % primes[j] == 0) {
                phi[primes[j] * i] = phi[i] * primes[j];    
                break;
            }
            phi[primes[j] * i] = phi[i] * (primes[j] - 1);
        }
    }
}
```

