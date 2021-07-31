# 快速幂

> 通过将幂二进制化进行简化运算时间复杂度

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

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

int main()
{
    int n;
    scanf("%d", &n);
    while (n -- ) {
        LL a , b , p;
        scanf ("%lld%lld%lld", &a , &b, &p);
        printf("%lld\n" , qmi (a , b , p));
    }
    return 0;
}
```

