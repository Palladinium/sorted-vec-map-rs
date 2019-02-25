# sorted_vec_map
A drop-in replacement for `BTreeMap`/`BTreeSet` backed by a `Vec` that exposes the same API and enforces the same invariants, while also allowing (immutable) access to the underlying continuous storage.

**This library is experimental.**
Currently the only tests are copied straight from the `BTreeMap`/`BTreeSet` implementations in the standard library.

# Complexity

 Data structure               | Indexing | Lookup    | Insertion | Deletion  |
------------------------------|----------|-----------|-----------|-----------|
`SortedVecMap`/`SortedVecSet` | O(1)     | O(log(n)) | O(n)      | O(n)      |
`BTreeMap`/`BTreeSet`         | O(n) (*) | O(log(n)) | O(log(n)) | O(log(n)) |

(*) `BTreeMap` can support O(log(n)) indexing but it is currently not implemented. See also [this thread on internals.rust-lang.org](https://internals.rust-lang.org/t/suggestion-btreemap-btreeset-o-log-n-n-th-element-access/9515)

# License

`sorted_vec_map` is licensed under the MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT)

# TODO list
- Nightly feature gate for APIs that are nightly-only for the std counterparts
- Benchmarks. It's likely that the stdlib collections will outperform this.
- Currently all searches are binary searches. Linear searches will likely outperform them for small sizes - bench
- Some Iterator trait impls are probably missing
- `ArrayVec` versions
