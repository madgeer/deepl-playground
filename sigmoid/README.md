# Logistic Regression from Scratch — Sigmoid, BCE, Softmax

Lab Minggu 3 Deep Learning. Dua file: `logistic.py` (implementasi) + `check.py` (verifikasi).

Jalankan:

```powershell
py .\check.py
```

## 1. `logistic.py` — apa isinya

| Fungsi | Rumus / Ide | Catatan numerik |
|---|---|---|
| `sigmoid(z)` | `1/(1+exp(-z))` untuk `z>=0`, `exp(z)/(1+exp(z))` untuk `z<0` | Cabang dua sisi agar `exp` tidak pernah menerima bilangan positif besar → anti-overflow. |
| `bce_loss(p, y)` | `-mean(y·log p + (1-y)·log(1-p))` | `clip` ke `[1e-15, 1-1e-15]` agar `log(0)` tidak terjadi. Pembagi `1/N` membuat `eta` portabel antar batch-size. |
| `softmax_naive(Z)` | `exp(Z) / sum(exp(Z))` persis seperti di slide | Sengaja dibiarkan overflow untuk demo B.4. `exp(800) = inf` di `float64` (batas ~`exp(709)`), lalu `inf/inf = nan`. |
| `softmax_stable(Z)` | `exp(Z-max) / sum(exp(Z-max))` per baris | Memakai invariansi geser `softmax(z+c) = softmax(z)`. Kurangi `max` baris → argumen `exp` ≤ 0 → tidak overflow. |
| `ce_loss(P, Y_idx)` | `-mean(log P[i, Y_idx[i]])` | `Y_idx` berisi indeks kelas, bukan one-hot. Sama seperti BCE, ada `clip` + `mean`. |
| `grads_binary(X, y, w, b)` | `dw = Xᵀ(p-y)/N`, `db = mean(p-y)` | Konvensi: baris = contoh (lihat A.9). `p = sigmoid(X@w + b)`. |
| `finite_diff(f, theta, h=1e-5)` | Central difference `(f(θ+h)-f(θ-h))/2h` per elemen | Mutasi `theta` in-place lalu dikembalikan. Hanya untuk cek gradien, bukan training. |

## 2. `check.py` — apa yang diverifikasi

### a. Zero-init check → harus `log 2` dan `log 10`

```python
w, b = zeros → sigmoid = 0.5 → BCE = -log(0.5) = log 2 ≈ 0.693147181
W, bb = zeros → softmax = 1/10 → CE = -log(1/10) = log 10 ≈ 2.302585093
```

Independen dari data/seed. Kalau tidak sama persis, berarti `mean` / `clip` / `softmax` salah.

### b. Gradient check → harus ~`1e-11`

Bandingkan gradien analitik (`grads_binary`) vs numerik (`finite_diff`):

```
max |analytic - numeric| = 1.284e-11
```

Orde `1e-11` = cocok sampai presisi `float64`. Kalau orde `1e-3` ke atas → rumus gradien salah.

### c. Overflow demo → `naive` gagal, `stable` benar

```python
Z = [[800.0, 799.0, 1.0]]
naive : [[nan nan  0.]]              # exp(800)=inf, inf/inf=nan + RuntimeWarning
stable: [[0.73105858 0.26894142 0.]] # Z-max = [[0,-1,-799]] → exp = [1, 0.368, 0]
```

`exp(-799)` *underflow* ke `0`, tapi itu benar sampai presisi mesin (nilainya ~`1e-347`). Jumlah baris stabil = 1.

## 3. Output yang diharapkan

```
zero-init BCE = 0.693147181 (log 2 = 0.693147181)
zero-init CE (K=10) = 2.302585093 (log 10 = 2.302585093)
max |analytic - numeric| = 1.284e-11
naive : [[nan nan  0.]]
stable: [[0.73105858 0.26894142 0.        ]]
```

Dua `RuntimeWarning: overflow/invalid` dari `softmax_naive` adalah normal — itulah yang ingin ditunjukkan.

## 4. Pelajaran kunci

1. Jangan pernah `exp` logit mentah — selalu kurangi `max` per baris dulu.
2. `sigmoid` juga butuh cabang stabil, bukan sekadar `1/(1+exp(-z))`.
3. Selalu `mean` loss (bukan `sum`) agar learning rate tidak tergantung ukuran batch.
4. Cek `log K` saat inisialisasi nol + gradient check dengan finite difference sebelum training.
