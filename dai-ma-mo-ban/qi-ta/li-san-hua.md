# 离散化

## 点离散

> 对于区间端点或者数值大的点进行离散化

```cpp
vector<int> alls;

int get(int x) {
    return lower_bound(alls.begin(), alls.end(), x) - alls.begin() + 1;
}

int main() {
    ...
    alls.push_back(l);
    alls.push_back(r);
    ...
    sort(alls.begin(), alls.end());
    alls.erase(unique(alls.begin(), alls.end()),alls.end());

    ...
    int l = get(l), r = get(r);
}
```

