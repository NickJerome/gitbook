# 数字三角形模型

> 即只能走下或者走右的移动路线模型



### [AcWing 1015. 摘花生](https://www.acwing.com/problem/content/description/1017/)

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100 + 10;

int n, m;
int dp[N][N];

void solve() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) {
        for (int j = 1; j <= m; ++ j) {
            scanf("%d", &dp[i][j]);
        }
    }
    
    for (int i = 1; i <= n; ++ i) {
        for (int j = 1; j <= m; ++ j) {
            dp[i][j] += max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    
    printf("%d\n", dp[n][m]);
}

int main() {
    int T;
    scanf("%d", &T);
    while (T --) {
        solve();
    }
    return 0;
}
```



### [AcWing 1018. 最低通行费](https://www.acwing.com/activity/content/problem/content/1257/1/)

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100 + 10;

int n;
int dp[N][N];

void solve() {
    scanf("%d", &n);
    
    memset(dp, 0x3f, sizeof dp);
    
    for (int i = 1; i <= n; ++ i) {
        for (int j = 1; j <= n; ++ j) {
            scanf("%d", &dp[i][j]);
        }
    }
    dp[0][1] = 0;
    for (int i = 1; i <= n; ++ i) {
        for (int j = 1; j <= n; ++ j) {
            dp[i][j] += min(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    
    printf("%d\n", dp[n][n]);
}

int main() {
    solve();
    return 0;
}
```



### [AcWing 1027. 方格取数](https://www.acwing.com/activity/content/problem/content/1258/1/)

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 100 + 10;

int n;
int w[N][N];
int dp[2 * N][N][N];

void solve() {
    scanf("%d", &n);
    
    int a, b, c;
    while (cin >> a >> b >> c, a || b || c) w[a][b] = c;
    
    for (int k = 2; k <= n + n; ++ k) {
        for (int i1 = 1; i1 <= n; ++ i1) {
            for (int i2 = 1; i2 <= n; ++ i2) {
                int j1 = k - i1, j2 = k - i2;
                if (1 <= j1 && j1 <= n && 1 <= j2 && j2 <= n) {
                    int t = w[i1][j1];
                    if (i1 != i2) t += w[i2][j2];
                    int &x = dp[k][i1][i2];
                    x = max(x, dp[k - 1][i1 - 1][i2 - 1] + t);
                    x = max(x, dp[k - 1][i1 - 1][i2] + t);
                    x = max(x, dp[k - 1][i1][i2 - 1] + t);
                    x = max(x, dp[k - 1][i1][i2] + t);
                }
            }
        }
    }
    
    printf("%d\n", dp[n + n][n][n]);
}

int main() {
    solve();
    return 0;
}
```



### [AcWing 275. 传纸条](https://www.acwing.com/problem/content/277/)

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 50 + 10;

int n, m;
int w[N][N];
int dp[2 * N][N][N];

void solve() {
    scanf("%d%d", &n, &m);
    
    for (int i = 1; i <= n; ++ i) {
        for (int j = 1; j <= m; ++ j) {
            scanf("%d", &w[i][j]);
        }
    }
    
    for (int k = 2; k <= n + m; ++ k) {
        for (int i1 = 1; i1 <= n; ++ i1) {
            for (int i2 = 1; i2 <= n; ++ i2) {
                int j1 = k - i1, j2 = k - i2;
                if (1 <= j1 && j1 <= m && 1 <= j2 && j2 <= m) {
                    int t = w[i1][j1];
                    if (i1 != i2) t += w[i2][j2];
                    int &x = dp[k][i1][i2];
                    x = max(x, dp[k - 1][i1 - 1][i2 - 1] + t);
                    x = max(x, dp[k - 1][i1 - 1][i2] + t);
                    x = max(x, dp[k - 1][i1][i2 - 1] + t);
                    x = max(x, dp[k - 1][i1][i2] + t);
                }
            }
        }
    }
    
    printf("%d\n", dp[n + m][n][n]);
}

int main() {
    solve();
    return 0;
}
```

