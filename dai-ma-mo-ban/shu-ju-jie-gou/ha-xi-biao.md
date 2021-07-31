# 哈希表

> 拉链法hash

```cpp
const int N = 100000 + 10;

struct hash_map {
    int h[N] , n[N] , v[N] , cnt;
    hash_map () { 
        for (int i = 0; i < N ; ++ i) h[i] = -1; 
        cnt = 0; 
    }
    int f (int x) { return (x % N + N) % N; }
    int size () { return cnt; }
    bool find (int x) {
        int k = f(x);
        for (int i = h[k] ; ~i ; i = n[i])
            if (v[i] == x)
                return true;
        return false;
    }
    bool insert (int x) {
        int k = f (x);
        for (int i = h[k] ; ~i ; i = n[i])
            if (v[i] == x) return false;
        v[cnt] = x;
        n[cnt] = h[k];
        h[k] = cnt++;
        return true;
    }
};
```

