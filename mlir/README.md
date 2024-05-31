# Assignment 1b: Rectangular Loop Tiling and Arithmetic Intensity in MLIR

## Question 1
**Write an algorithm to compute Arithmetic Intensity of a perfectly nested affine loop in MLIR. Explain some of the limitations of your algorithm (if any) with simple examples.**

### Algorithm

1. **Initialize Variables**
   - Declare and Initialize variables to count arithmetic operations.

2. **Iterate Over Affine For Loops**
   - Use `getOperation()->walk` to traverse all the affine for loops in the MLIR operation.

3. **Handle Load and Store Operations:**
   -  Count the number of load and store operations inside each loop using `op->walk` with a lambda function for `affine::LoadOp` and `affine::StoreOp`.

4. **Count Arithmetic Operations**
   - Similarly, count various arithmetic operations (addition, subtraction, multiplication, division, etc.) inside the loop using `op->walk` with lambda functions for different arithmetic operations.

5. **Compute Iteration Count**
   - Calculate the total number of iterations for each loop by multiplying the differences between upper and lower loop bounds.

6. **Calculate Arithmetic Intensity**
   - Compute the total number of arithmetic operations (A) and the total number of memory access operations (B).
   - Compute the Arithmetic Intensity (AI) using the formula AI = (Number of Arithmetic Operations) / (Number of Memory Accesses), and print the result.

### Limitations
- Assumes perfectly nested affine loops. If loops are not perfectly nested, the algorithm might not correctly calculate the iteration count and AI.

- Counts only supported arithmetic operations (add, subtract, multiply, divide, remainder) and memory accesses (load, store). Custom or unsupported operations won't be considered.

## Question 2
**What is loop tiling and its impact on performance. Explain with reference to some examples, and also show change in runtime (in secs) on the matmul.mlir example shown above.**

Loop tiling, also known as loop blocking, is a technique used to improve cache locality and reduce memory access overhead in nested loops. It involves dividing the nesting of loops into smaller blocks or tiles.

This transformation allows data to be accessed in blocks (tiles), with the block size defined as a parameter of this transformation. Each loop is transformed in two loops: one iterating inside each block (intratile) and the other one iterating over the blocks (intertile).

`matmul.mlir`

```
func.func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>, %arg2: memref<?x300xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 200 {
      affine.for %arg4 = 0 to 300 {
        affine.for %arg5 = 0 to 400 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<?x400xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<?x300xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg2[%arg3, %arg4] : memref<?x300xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<?x300xf32>
        }
      }
    }
    return
  }
```

`tiled-mat-mul.mlir`

```
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 32, 200)>
#map2 = affine_map<(d0) -> (d0 + 32, 300)>
#map3 = affine_map<(d0) -> (d0 + 32, 400)>
module {
  func.func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>, %arg2: memref<?x300xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 200 step 32 {
      affine.for %arg4 = 0 to 300 step 32 {
        affine.for %arg5 = 0 to 400 step 32 {
          affine.for %arg6 = #map(%arg3) to min #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to min #map2(%arg4) {
              affine.for %arg8 = #map(%arg5) to min #map3(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<?x400xf32>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<?x300xf32>
                %2 = arith.mulf %0, %1 : f32
                %3 = affine.load %arg2[%arg6, %arg7] : memref<?x300xf32>
                %4 = arith.addf %3, %2 : f32
                affine.store %4, %arg2[%arg6, %arg7] : memref<?x300xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
```

Applying loop tiling to the `matmul.mlir` example can improve cache utilization, reduce cache misses, and enhance data reuse, resulting in faster execution time.

## Question 2
**What is the legality condition for loop tiling? Explain briefly with an example.**

In terms of legality, loop tiling is generally legal as long as it preserves the original semantics of the program and does not introduce any unintended side effects or changes in behavior. 

For example, loop tiling should not alter the order of computations or the final results of the program.

Suppose we have a nested loop that performs matrix multiplication

```
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

### Incorrect Loop Tiling (Alters Computation Order)

```
for (int i = 0; i < N; i += T) {
    for (int j = 0; j < N; j += T) {
        for (int k = 0; k < N; k += T) {
            for (int kk = k; kk < k + T; kk++) {
                for (int ii = i; ii < i + T; ii++) {
                    for (int jj = j; jj < j + T; jj++) {
                        C[ii][jj] += A[ii][kk] * B[kk][jj];
                    }
                }
            }
        }
    }
}
```

In this incorrect transformation, the order of computation within the innermost loop is changed. Specifically, the order of indices ii and kk is swapped. This alteration results in incorrect matrix multiplication and violates the requirement that loop tiling should not change the order of computations.

To correct this, we need to ensure that the order of computations remains unchanged during loop tiling. Here's the corrected version of loop tiling for matrix multiplication:

### Corrected Loop Tiling:

```
for (int i = 0; i < N; i += T) {
    for (int j = 0; j < N; j += T) {
        for (int k = 0; k < N; k += T) {
            // Compute each tile
            for (int ii = i; ii < i + T; ii++) {
                for (int jj = j; jj < j + T; jj++) {
                    for (int kk = k; kk < k + T; kk++) {
                        C[ii][jj] += A[ii][kk] * B[kk][jj];
                    }
                }
            }
        }
    }
}
```

In the corrected version, the order of computations within the innermost loop `(ii, jj, kk)` remains the same as in the original code, ensuring that the matrix multiplication is performed correctly after loop tiling.

Specifically, for loop tiling to be legal, the following conditions typically need to hold.

### Conditions

- **Dependency Preservation:** Tiling should not introduce dependencies between iterations that didn't exist in the original loop nest. For example, consider a loop nest where elements in a tile depend on elements outside the tile. Tiling such a loop may introduce incorrect dependencies.

- **Boundary Conditions:** Tiling should correctly handle boundary conditions and edge cases. For instance, when tiling a loop, ensuring correct handling of loop bounds and boundary conditions is crucial to maintain correctness.

- **Data Alignment:** Tiling should preserve data alignment and access patterns. If the original loop nest had aligned memory accesses, tiling should not disrupt this alignment, ensuring correct data accesses after tiling.

Ensuring these conditions maintains program correctness after loop tiling.