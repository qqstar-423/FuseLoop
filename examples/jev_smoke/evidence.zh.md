# Chinese-language preparation material: for translation and connectivity testing only

The following is a synthetic record; it was not executed on an NPU and does not represent real operator performance.

The evaluation protocol follows cann-bench's kernel_details: it aggregates the median execution time of each kernel and excludes gaps between kernels. baseline, speedup, HAP and the total score all use the tool's original values. The synthetic profiling conclusion: writing back and reading intermediate results may create extra traffic; this is not sufficient to prove a single kernel is faster — other fusion plans need to be measured under the same protocol.

| Round | Optimization direction | Implementation change | Geomean speedup | Min case speedup | Outcome |
|---|---|---|---|---|---|
| 1 | Reduce duplicate reads | Reuse already-loaded weight tiles | 1.10 | 1.01 | Accepted, first compliance |
| 2 | Improve tiling | Increase tile reuse rate | 1.13 | 1.02 | Accepted, new best version |
| 3 | Reduce synchronization | Remove unnecessary barriers | 1.14 | 1.02 | Accepted, but small gain |
| 4 | Reduce temporary storage | Reuse scratch space | 1.08 | 1.00 | Rejected, rolled back and the regression recorded |
| 5 | Adjust Vector tiling | Modify Vector block size based on the restored best version | 1.15 | 1.03 | Accepted, becomes the current version |
