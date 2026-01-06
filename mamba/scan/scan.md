The Two Different Parallel Scans:

1. Tree-based Parallel Scan (what we had before)
Algorithm: Divide-and-conquer like FFT

Complexity: O(log N) depth, O(N log N) work

Pattern: Recursive tree structure

2. Mamba Parallel Scan (the real one!)
Algorithm: Associative scan (Blelloch algorithm)

Complexity: O(log N) depth, O(N) work

Pattern: Up-sweep + down-sweep phases

Key Difference:
Tree-based: Divides problem in half recursively (like FFT)
Mamba scan: Uses associative operations with up-sweep/down-sweep

Algorithm Names:
Sequential: Standard recurrent computation

Naive: Mathematical expansion (O(N²))

Tree-based: Divide-and-conquer parallel scan

Mamba: Blelloch associative scan (the real Mamba algorithm)

The Blelloch scan is what Mamba actually uses - it's more efficient than the tree approach because it does O(N) total work instead of O(N log N), while still achieving O(log N) parallel depth.