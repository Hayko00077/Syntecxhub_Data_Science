"""
============================================================
  Syntecxhub Internship — Data Science
  Project 1: NumPy Data Explorer
============================================================
"""

import numpy as np
import time
import os

# ────────────────────────────────────────────────────────
# 1. ARRAY CREATION (Զանգվածի ստեղծում)
# ────────────────────────────────────────────────────────
print("=" * 55)
print("  1. ARRAY CREATION")
print("=" * 55)

arr1d = np.array([10, 20, 30, 40, 50])
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

zeros   = np.zeros((3, 3))
ones    = np.ones((2, 4))
rng     = np.arange(0, 20, 2)
linsp   = np.linspace(0, 1, 6)
random  = np.random.randint(1, 100, size=(4, 4))

print(f"1D Array        : {arr1d}")
print(f"2D Array:\n{arr2d}")
print(f"Zeros (3x3):\n{zeros}")
print(f"Ones  (2x4):\n{ones}")
print(f"Arange (0-18,2) : {rng}")
print(f"Linspace (0-1)  : {linsp}")
print(f"Random (4x4):\n{random}")


# ────────────────────────────────────────────────────────
# 2. INDEXING & SLICING (Ինդեքսավորում և կտրատում)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  2. INDEXING & SLICING")
print("=" * 55)

data = np.array([[10, 20, 30, 40],
                 [50, 60, 70, 80],
                 [90, 100, 110, 120]])

print(f"Full array:\n{data}")
print(f"Element [1][2]       : {data[1][2]}")
print(f"First row            : {data[0]}")
print(f"Last column          : {data[:, -1]}")
print(f"Rows 0-1, Cols 1-2:\n{data[0:2, 1:3]}")
print(f"Boolean (values>50)  : {data[data > 50]}")


# ────────────────────────────────────────────────────────
# 3. MATHEMATICAL OPERATIONS (Մաթեմատիկական գործողություններ)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  3. MATHEMATICAL OPERATIONS")
print("=" * 55)

a = np.array([10, 20, 30, 40, 50])
b = np.array([2,  4,  6,  8,  10])

print(f"a          : {a}")
print(f"b          : {b}")
print(f"a + b      : {a + b}")
print(f"a - b      : {a - b}")
print(f"a * b      : {a * b}")
print(f"a / b      : {a / b}")
print(f"a ** 2     : {a ** 2}")
print(f"sqrt(a)    : {np.sqrt(a)}")
print(f"dot(a, b)  : {np.dot(a, b)}")


# ────────────────────────────────────────────────────────
# 4. AXIS-WISE OPERATIONS (Առանցքային գործողություններ)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  4. AXIS-WISE OPERATIONS")
print("=" * 55)

matrix = np.array([[3, 7, 1],
                   [9, 2, 8],
                   [4, 6, 5]])

print(f"Matrix:\n{matrix}")
print(f"Sum (all)        : {np.sum(matrix)}")
print(f"Sum (axis=0 col) : {np.sum(matrix, axis=0)}")
print(f"Sum (axis=1 row) : {np.sum(matrix, axis=1)}")
print(f"Min (axis=0)     : {np.min(matrix, axis=0)}")
print(f"Max (axis=1)     : {np.max(matrix, axis=1)}")
print(f"Cumsum (all)     : {np.cumsum(matrix)}")


# ────────────────────────────────────────────────────────
# 5. STATISTICAL OPERATIONS (Վիճակագրական գործողություններ)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  5. STATISTICAL OPERATIONS")
print("=" * 55)

dataset = np.array([23, 45, 12, 67, 34, 89, 56, 78, 90, 11,
                    44, 55, 33, 77, 22, 88, 66, 99, 10, 50])

print(f"Dataset  : {dataset}")
print(f"Mean     : {np.mean(dataset):.2f}")
print(f"Median   : {np.median(dataset):.2f}")
print(f"Std Dev  : {np.std(dataset):.2f}")
print(f"Variance : {np.var(dataset):.2f}")
print(f"Min      : {np.min(dataset)}")
print(f"Max      : {np.max(dataset)}")
print(f"25th pct : {np.percentile(dataset, 25):.2f}")
print(f"75th pct : {np.percentile(dataset, 75):.2f}")


# ────────────────────────────────────────────────────────
# 6. RESHAPING & BROADCASTING
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  6. RESHAPING & BROADCASTING")
print("=" * 55)

original = np.arange(1, 13)
print(f"Original (12,)    : {original}")

r1 = original.reshape(3, 4)
print(f"Reshaped (3x4):\n{r1}")

r2 = original.reshape(2, 2, 3)
print(f"Reshaped (2x2x3):\n{r2}")

flat = r1.flatten()
print(f"Flattened         : {flat}")

# Broadcasting
col_vec = np.array([[10], [20], [30]])
row_vec = np.array([1, 2, 3, 4])
broadcast_result = col_vec + row_vec
print(f"\nBroadcasting col+row:\n{broadcast_result}")

# Transpose
t = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nOriginal (2x3):\n{t}")
print(f"Transposed (3x2):\n{t.T}")


# ────────────────────────────────────────────────────────
# 7. SAVE & LOAD (Պահպանում և բեռնում)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  7. SAVE & LOAD OPERATIONS")
print("=" * 55)

save_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Save single array (.npy)
np.save("my_array.npy", save_arr)
loaded = np.load("my_array.npy")
print(f"Saved & Loaded (.npy):\n{loaded}")

# Save multiple arrays (.npz)
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.savez("multi_arrays.npz", x=x, y=y)
loaded_npz = np.load("multi_arrays.npz")
print(f"Loaded x from .npz : {loaded_npz['x']}")
print(f"Loaded y from .npz : {loaded_npz['y']}")

# Save as CSV (txt)
np.savetxt("array.csv", save_arr, delimiter=",", fmt="%d")
loaded_csv = np.loadtxt("array.csv", delimiter=",")
print(f"Saved & Loaded (.csv):\n{loaded_csv}")

# Cleanup
for f in ["my_array.npy", "multi_arrays.npz", "array.csv"]:
    if os.path.exists(f):
        os.remove(f)
print("Temp files cleaned up.")


# ────────────────────────────────────────────────────────
# 8. NUMPY vs PYTHON LIST — PERFORMANCE COMPARISON
# ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  8. NUMPY vs PYTHON LIST — PERFORMANCE")
print("=" * 55)

N = 1_000_000

# Python list
py_list = list(range(N))
start = time.time()
py_result = [x * 2 for x in py_list]
py_time = time.time() - start

# NumPy array
np_arr = np.arange(N)
start = time.time()
np_result = np_arr * 2
np_time = time.time() - start

print(f"Size             : {N:,} elements")
print(f"Python list time : {py_time:.4f} sec")
print(f"NumPy array time : {np_time:.4f} sec")
print(f"NumPy is {py_time / np_time:.1f}x faster than Python list!")

print("\n" + "=" * 55)
print("  NumPy Data Explorer — COMPLETED ✓")
print("=" * 55)