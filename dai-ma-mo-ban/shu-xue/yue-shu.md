# 约数

> 字面意思

## 试除法求约束

```cpp
vector<int> get_divisors (int x) {
    vector<int> res;
    for (int i = 1; i <= x / i; ++ i) {
        if (x % i == 0) {
            res.push_back (i);
            if (i != x / i) res.push_back(x / i);
        }
    }
    sort (res.begin(), res.end());
    return res;
}
```

## 约数个数

> 设一个数的质因子为$p_{i}$，幂为$b_{i}$
>
> 则这个的个数应该有$\prod_{i=1}^n \( b_{i} + 1 \)$

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>
#include <unordered_map>
typedef long long ll;

using namespace std;

const int MOD = 1e9 + 7;

unordered_map<int, int> factors;

int main () {
    int n;
    ll res = 1;
    scanf("%d", &n);
    for (int i = 1; i <= n; ++ i) {
        int x;
        scanf ("%d", &x);
        for (int j = 2; j <= x / j; ++ j) {
            while (x % j == 0) {
                x /= j;
                factors[j]++;
            }
        }
        if (x > 1) factors[x]++;
    }
    for (auto factor : factors) res = res * (factor.second + 1) % MOD;

    printf ("%lld", res);

    return 0;
}
```

## 约数和

> $res = \(p\_1^0+p\_1^1+…+p\_1^{c\_1}\)∗…∗\(p\_k^0+p\_k^1+…+p\_k^{c\_k}\)$ 乘出来就知道了

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>
#include <unordered_map>

using namespace std;

typedef long long ll;

const int N = 100 + 10;
const int MOD = 1e9 + 7;

unordered_map <int,int> factors;

ll qmi (ll a, ll b, ll p) {
    ll res = 1 % p;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return res % p;
}

int main () {
    int n;
    scanf ("%d", &n);
    while (n --) {
        int x;
        scanf ("%d", &x);

        for (int i = 2; i <= x / i; ++ i) {
            while (x % i == 0) {
                x /= i;
                factors[i]++;
            }
        }
        if (x > 1) factors[x]++;
    }

    ll res = 1;
    for (auto factor : factors) {
        ll p = 1, k = factor.second;
        while (k --) p = (p * factor.first + 1)% MOD;
        res = res * p % MOD;
    }

    printf ("%lld", res);

    return 0;
}
```

