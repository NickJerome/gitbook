# 素数

> 即只有1和自己本身能整除

## 素数判定

```cpp
bool is_prime(int x) 
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}
```

## 分解质因数

```cpp
void divide (int x) {
    for (int i = 2 ; i <= x / i ; ++ i) {
        if (x % i == 0) {
            int t = 0;
            while (x % i == 0) x /= i , t ++;
            printf ("%d %d\n" , i , t);
        }
    }
    if (x > 1) printf ("%d %d\n" , x , 1);
}
```

## 筛素数

### 欧拉筛

> 每一个数只会被它的最小质因子筛去

```cpp
    const int N = 1000000 + 10;

int p[N] , v[N];

bool get_prime (int n) {
    p[0] = 0;
    v[1] = 1;
    for (int i = 2 ; i <= n ; ++ i) {
        if (v[i] == 0) p[++p[0]] = i;

        for (int j = 1 ; j <= p[0] && i * p[j] <= n ; ++ j) {
            v[i * p[j]] = 1;
            if (i % p[j] == 0) break;
        }
    }
}
```

