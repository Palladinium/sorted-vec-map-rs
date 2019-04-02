use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{self, Debug},
    hash::{self, Hash},
    iter::*,
    marker::PhantomData,
    mem,
    ops::{
        Bound::{self, *},
        Index, RangeBounds,
    },
    slice, vec,
};

use serde::{
    de::{Deserialize, Deserializer, MapAccess, Visitor},
    ser::{Serialize, SerializeMap, Serializer},
};

/// A structure similar to `BTreeMap`, implemented in terms of a sorted `Vec<K, V>`.
/// The API of this struct will attempt to be, by convention, as compatible to `BTreeMap` as possible.
pub struct SortedVecMap<K, V> {
    vec: Vec<(K, V)>,
}

impl<K, V> Debug for SortedVecMap<K, V>
where
    K: Debug + Ord,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Ord, V> Default for SortedVecMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for SortedVecMap<K, V> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.vec.eq(&rhs.vec)
    }
}

impl<K: Eq, V: Eq> Eq for SortedVecMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for SortedVecMap<K, V> {
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.vec.partial_cmp(&rhs.vec)
    }
}

impl<K: Ord, V: Ord> Ord for SortedVecMap<K, V> {
    #[inline]
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.vec.cmp(&rhs.vec)
    }
}

impl<K: Ord, V> SortedVecMap<K, V> {
    #[inline]
    pub fn from_vec(mut vec: Vec<(K, V)>) -> Self {
        vec.sort_unstable_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        vec.dedup_by(|lhs, rhs| lhs.0.eq(&rhs.0));

        Self { vec }
    }

    // Replicated `Vec<(K, V)>` APIs follow
    #[inline]
    pub fn new() -> Self {
        Self { vec: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.vec.reserve(additional)
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.vec.reserve_exact(additional)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit()
    }

    #[inline]
    pub fn into_vec(self) -> Vec<(K, V)> {
        self.vec
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.vec.truncate(len)
    }

    #[inline]
    pub fn as_slice(&self) -> &[(K, V)] {
        self.vec.as_slice()
    }

    #[inline]
    pub fn remove_at(&mut self, index: usize) -> (K, V) {
        self.vec.remove(index)
    }

    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut((&K, &V)) -> bool,
    {
        self.vec.retain(|e| f((&e.0, &e.1)))
    }

    #[inline]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, K, V>
    where
        R: RangeBounds<usize>,
    {
        self.vec.drain(range)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.vec.clear()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    #[inline]
    pub fn split_off_at(&mut self, at: usize) -> SortedVecMap<K, V> {
        Self {
            vec: self.vec.split_off(at),
        }
    }

    // Replicated `BTreeMap<K, V>` APIs follow

    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.vec
            .binary_search_by(|item| item.0.borrow().cmp(key))
            .ok()
            .map(|idx| &self.vec[idx].1)
    }

    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.vec
            .binary_search_by(|item| item.0.borrow().cmp(key))
            .is_ok()
    }

    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.vec
            .binary_search_by(|item| item.0.borrow().cmp(key))
            .ok()
            .map(move |idx| &mut self.vec[idx].1)
    }

    #[inline]
    pub fn insert(&mut self, key: K, mut value: V) -> Option<V> {
        match self.vec.binary_search_by(|item| item.0.cmp(&key)) {
            Ok(idx) => {
                mem::swap(&mut self.vec[idx].1, &mut value);
                Some(value)
            }
            Err(idx) => {
                self.vec.insert(idx, (key, value));
                None
            }
        }
    }

    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.vec
            .binary_search_by(|item| item.0.borrow().cmp(key))
            .ok()
            .map(|idx| self.vec.remove(idx).1)
    }

    #[inline]
    pub fn append(&mut self, other: &mut SortedVecMap<K, V>) {
        self.vec.reserve(other.len());

        for (k, v) in other.vec.drain(..) {
            self.insert(k, v);
        }
    }

    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        match self.vec.binary_search_by(|item| item.0.cmp(&key)) {
            Ok(idx) => Entry::Occupied(OccupiedEntry {
                key,
                idx,
                vec: &mut self.vec,
            }),
            Err(idx) => Entry::Vacant(VacantEntry {
                key,
                idx,
                vec: &mut self.vec,
            }),
        }
    }

    #[inline]
    pub fn split_off<Q>(&mut self, key: &Q) -> SortedVecMap<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let idx = match self.vec.binary_search_by(|item| item.0.borrow().cmp(key)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        Self {
            vec: self.vec.split_off(idx),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.vec.iter(),
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            iter: self.vec.iter_mut(),
        }
    }

    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys {
            iter: self.vec.iter(),
        }
    }

    #[inline]
    pub fn values(&self) -> Values<'_, K, V> {
        Values {
            iter: self.vec.iter(),
        }
    }

    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            iter: self.vec.iter_mut(),
        }
    }

    #[inline]
    pub fn range<T, R>(&self, range: R) -> Range<'_, K, V>
    where
        K: Borrow<T>,
        R: RangeBounds<T>,
        T: Ord + ?Sized,
    {
        Range {
            iter: self.range_as_slice(range).iter(),
        }
    }

    #[inline]
    pub fn range_as_slice<T, R>(&self, range: R) -> &[(K, V)]
    where
        K: Borrow<T>,
        R: RangeBounds<T>,
        T: Ord + ?Sized,
    {
        let (start, end) = self.find_range(range);

        // Required because (&Bound, &Bound) doesn't impl SliceIndex
        // See also https://github.com/rust-lang/rust/issues/35729
        match (start, end) {
            (Included(s), Included(e)) => &self.vec[s..=e],
            (Included(s), Excluded(e)) => &self.vec[s..e],
            (Included(s), Unbounded) => &self.vec[s..],

            (Excluded(s), Included(e)) => &self.vec[(s + 1)..=e],
            (Excluded(s), Excluded(e)) => &self.vec[(s + 1)..e],
            (Excluded(s), Unbounded) => &self.vec[(s + 1)..],

            (Unbounded, Included(e)) => &self.vec[..=e],
            (Unbounded, Excluded(e)) => &self.vec[..e],
            (Unbounded, Unbounded) => &self.vec[..],
        }
    }

    #[inline]
    pub fn range_mut<T, R>(&mut self, range: R) -> RangeMut<'_, K, V>
    where
        K: Borrow<T>,
        R: RangeBounds<T>,
        T: Ord + ?Sized,
    {
        RangeMut {
            iter: self.range_as_mut_slice(range).iter_mut(),
        }
    }

    // Private because it would allow users to modify keys. Use `range_mut` instead.
    fn range_as_mut_slice<T, R>(&mut self, range: R) -> &mut [(K, V)]
    where
        K: Borrow<T>,
        R: RangeBounds<T>,
        T: Ord + ?Sized,
    {
        let (start, end) = self.find_range(range);

        // Required because (&Bound, &Bound) doesn't impl SliceIndex
        // See also https://github.com/rust-lang/rust/issues/35729
        match (start, end) {
            (Included(s), Included(e)) => &mut self.vec[s..=e],
            (Included(s), Excluded(e)) => &mut self.vec[s..e],
            (Included(s), Unbounded) => &mut self.vec[s..],

            (Excluded(s), Included(e)) => &mut self.vec[(s + 1)..=e],
            (Excluded(s), Excluded(e)) => &mut self.vec[(s + 1)..e],
            (Excluded(s), Unbounded) => &mut self.vec[(s + 1)..],

            (Unbounded, Included(e)) => &mut self.vec[..=e],
            (Unbounded, Excluded(e)) => &mut self.vec[..e],
            (Unbounded, Unbounded) => &mut self.vec[..],
        }
    }

    fn find_range<T, R>(&self, range: R) -> (Bound<usize>, Bound<usize>)
    where
        K: Borrow<T>,
        R: RangeBounds<T>,
        T: Ord + ?Sized,
    {
        match (range.start_bound(), range.end_bound()) {
            (Excluded(s), Excluded(e)) if s == e => {
                panic!("range start and end are equal and excluded in SortedVecMap")
            }
            (Included(s), Included(e))
            | (Included(s), Excluded(e))
            | (Excluded(s), Included(e))
            | (Excluded(s), Excluded(e))
                if s > e =>
            {
                panic!("range start is greater than range end in SortedVecMap")
            }
            _ => {}
        };

        let start = match range.start_bound() {
            Included(b) => match self.vec.binary_search_by(|item| item.0.borrow().cmp(b)) {
                Ok(idx) => Included(idx),
                Err(idx) => Included(idx),
            },
            Excluded(b) => match self.vec.binary_search_by(|item| item.0.borrow().cmp(b)) {
                Ok(idx) => Excluded(idx),
                Err(idx) => Included(idx),
            },
            Unbounded => Unbounded,
        };

        let end = match range.end_bound() {
            Included(b) => match self.vec.binary_search_by(|item| item.0.borrow().cmp(b)) {
                Ok(idx) => Included(idx),
                Err(idx) => Excluded(idx),
            },
            Excluded(b) => match self.vec.binary_search_by(|item| item.0.borrow().cmp(b)) {
                Ok(idx) => Excluded(idx),
                Err(idx) => Excluded(idx),
            },
            Unbounded => Unbounded,
        };

        (start, end)
    }

    #[inline]
    pub fn position<Q>(&mut self, key: &Q) -> Result<usize, usize>
    where
        Q: Ord + ?Sized,
        K: Borrow<Q>,
    {
        self.vec.binary_search_by(|item| item.0.borrow().cmp(key))
    }
}

type Drain<'a, K, V> = vec::Drain<'a, (K, V)>;

/// An iterator over the entries of a `SortedVecMap`.
///
/// This `struct` is created by the [`iter`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`iter`]: struct.SortedVecMap.html#method.iter
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug, Clone)]
pub struct Iter<'a, K, V> {
    iter: slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (&e.0, &e.1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| (&e.0, &e.1))
    }
}

/// A mutable iterator over the entries of a `SortedVecMap`.
///
/// This `struct` is created by the [`iter_mut`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`iter_mut`]: struct.SortedVecMap.html#method.iter_mut
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug)]
pub struct IterMut<'a, K, V> {
    iter: slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (&e.0, &mut e.1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| (&e.0, &mut e.1))
    }
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`SortedVecMap`].
///
/// [`SortedVecMap`]: struct.SortedVecMap.html
/// [`entry`]: struct.SortedVecMap.html#method.entry
pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Vacant(vacant) => vacant.insert(default),
            Entry::Occupied(occupied) => occupied.into_mut(),
        }
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Vacant(vacant) => vacant.insert(default()),
            Entry::Occupied(occupied) => occupied.into_mut(),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Vacant(vacant) => vacant.key(),
            Entry::Occupied(occupied) => occupied.key(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Entry<'a, K, V>
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Vacant(vacant) => Entry::Vacant(vacant),
            Entry::Occupied(mut occupied) => {
                f(occupied.get_mut());
                Entry::Occupied(occupied)
            }
        }
    }
}

impl<'a, K: Ord, V: Default> Entry<'a, K, V> {
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Vacant(vacant) => vacant.insert(V::default()),
            Entry::Occupied(occupied) => occupied.into_mut(),
        }
    }
}

impl<'a, K: 'a + fmt::Debug + Ord, V: 'a + fmt::Debug> fmt::Debug for Entry<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Entry::Vacant(ref v) => f.debug_tuple("Entry").field(v).finish(),
            Entry::Occupied(ref o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// A view into a vacant entry in a `SortedVecMap`.
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
pub struct VacantEntry<'a, K, V> {
    key: K,
    vec: &'a mut Vec<(K, V)>,
    idx: usize,
}

impl<'a, K: 'a + fmt::Debug + Ord, V: 'a> fmt::Debug for VacantEntry<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        self.vec.insert(self.idx, (self.key, value));
        &mut self.vec[self.idx].1
    }
}

/// A view into an occupied entry in a `SortedVecMap`.
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
pub struct OccupiedEntry<'a, K, V> {
    key: K,
    vec: &'a mut Vec<(K, V)>,
    idx: usize,
}

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        self.vec.remove(self.idx)
    }

    #[inline]
    pub fn get(&self) -> &V {
        &self.vec[self.idx].1
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.vec[self.idx].1
    }

    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        &mut self.vec[self.idx].1
    }

    #[inline]
    pub fn insert(&mut self, mut value: V) -> V {
        mem::swap(&mut self.vec[self.idx].1, &mut value);
        value
    }

    #[inline]
    pub fn remove(self) -> V {
        self.vec.remove(self.idx).1
    }
}

impl<'a, K: 'a + fmt::Debug + Ord, V: 'a + fmt::Debug> fmt::Debug for OccupiedEntry<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<K: Ord + Clone, V: Clone> SortedVecMap<K, V> {
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[(K, V)]) {
        for (k, v) in other {
            self.insert(k.clone(), v.clone());
        }
    }
}

/// An iterator over the keys of a `SortedVecMap`.
///
/// This `struct` is created by the [`keys`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`keys`]: struct.SortedVecMap.html#method.keys
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug, Clone)]
pub struct Keys<'a, K, V> {
    iter: slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| &item.0)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|item| &item.0)
    }
}

/// An iterator over the values of a `SortedVecMap`.
///
/// This `struct` is created by the [`values`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`values`]: struct.SortedVecMap.html#method.values
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug, Clone)]
pub struct Values<'a, K, V> {
    iter: slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| &item.1)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> FusedIterator for Values<'a, K, V> {}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|item| &item.1)
    }
}

/// A mutable iterator over the values of a `SortedVecMap`.
///
/// This `struct` is created by the [`values_mut`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`values_mut`]: struct.SortedVecMap.html#method.values_mut
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug)]
pub struct ValuesMut<'a, K, V> {
    iter: slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| &mut item.1)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V> {}

impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V> {}

impl<'a, K, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|item| &mut item.1)
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for SortedVecMap<K, V> {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        Self::from_vec(Vec::from_iter(iter))
    }
}

impl<K: Ord + Clone, V: Clone> Clone for SortedVecMap<K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            vec: self.vec.clone(),
        }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.vec.clone_from(&other.vec)
    }
}

impl<K: Ord + Hash, V: Hash> Hash for SortedVecMap<K, V> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vec.hash(state)
    }
}

/// An owning iterator over the entries of a `SortedVecMap`.
///
/// This `struct` is created by the [`into_iter`] method on [`SortedVecMap`][`SortedVecMap`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: struct.SortedVecMap.html#method.into_iter
/// [`SortedVecMap`]: struct.SortedVecMap.html
pub struct IntoIter<K, V> {
    iter: vec::IntoIter<(K, V)>,
}

impl<K: Ord, V> IntoIterator for SortedVecMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.vec.into_iter(),
        }
    }
}

impl<K: Ord, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K: Ord, V> DoubleEndedIterator for IntoIter<K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<K: Ord, V> ExactSizeIterator for IntoIter<K, V> {}

impl<K: Ord, V> FusedIterator for IntoIter<K, V> {}

impl<'a, K: Ord, V> IntoIterator for &'a mut SortedVecMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a SortedVecMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator over a sub-range of entries in a `SortedVecMap`.
///
/// This `struct` is created by the [`range`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`range`]: struct.SortedVecMap.html#method.range
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug, Clone)]
pub struct Range<'a, K, V> {
    iter: slice::Iter<'a, (K, V)>,
}

impl<'a, K: Ord, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (&e.0, &e.1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// A mutable iterator over a sub-range of entries in a `SortedVecMap`.
///
/// This `struct` is created by the [`range_mut`] method on [`SortedVecMap`]. See its
/// documentation for more.
///
/// [`range_mut`]: struct.SortedVecMap.html#method.range_mut
/// [`SortedVecMap`]: struct.SortedVecMap.html
#[derive(Debug)]
pub struct RangeMut<'a, K, V> {
    iter: slice::IterMut<'a, (K, V)>,
}

impl<'a, K: Ord, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (&e.0, &mut e.1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, Q, V> Index<&'a Q> for SortedVecMap<K, V>
where
    K: Ord + Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `SortedVecMap`.
    #[inline]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<'a, K, V> Extend<(&'a K, &'a V)> for SortedVecMap<K, V>
where
    K: Copy + Ord,
    V: Copy,
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K, V> Extend<(K, V)> for SortedVecMap<K, V>
where
    K: Ord,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Ord, V> Into<Vec<(K, V)>> for SortedVecMap<K, V> {
    #[inline]
    fn into(self) -> Vec<(K, V)> {
        self.into_vec()
    }
}

impl<K, V> Serialize for SortedVecMap<K, V>
where
    K: Ord + Serialize,
    V: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;

        for (k, v) in self.iter() {
            map.serialize_entry(k, v)?;
        }

        map.end()
    }
}

impl<'de, K, V> Deserialize<'de> for SortedVecMap<K, V>
where
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(SortedVecMapVisitor::new())
    }
}

struct SortedVecMapVisitor<K, V>(PhantomData<(K, V)>);

impl<K, V> SortedVecMapVisitor<K, V> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'de, K, V> Visitor<'de> for SortedVecMapVisitor<K, V>
where
    K: Ord + Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = SortedVecMap<K, V>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a map")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut map = SortedVecMap::with_capacity(access.size_hint().unwrap_or(0));

        // While there are entries remaining in the input, add them
        // into our map.
        while let Some((key, value)) = access.next_entry()? {
            map.insert(key, value);
        }

        Ok(map)
    }
}

#[macro_export]
macro_rules! sorted_vec_map {
    ($($k:expr => $v:expr),*) => { SortedVecMap::from_vec(vec![$(($k, $v)),*])};
}

#[cfg(test)]
mod tests {
    use super::{
        super::test_utils::DeterministicRng,
        {Entry::*, *},
    };
    use std::ops::Bound::{self, Excluded, Included, Unbounded};
    use std::rc::Rc;

    #[test]
    fn test_basic_large() {
        let mut map = SortedVecMap::new();
        #[cfg(not(miri))] // Miri is too slow
        let size = 10000;
        #[cfg(miri)]
        let size = 200;
        assert_eq!(map.len(), 0);

        for i in 0..size {
            assert_eq!(map.insert(i, 10 * i), None);
            assert_eq!(map.len(), i + 1);
        }

        for i in 0..size {
            assert_eq!(map.get(&i).unwrap(), &(i * 10));
        }

        for i in size..size * 2 {
            assert_eq!(map.get(&i), None);
        }

        for i in 0..size {
            assert_eq!(map.insert(i, 100 * i), Some(10 * i));
            assert_eq!(map.len(), size);
        }

        for i in 0..size {
            assert_eq!(map.get(&i).unwrap(), &(i * 100));
        }

        for i in 0..size / 2 {
            assert_eq!(map.remove(&(i * 2)), Some(i * 200));
            assert_eq!(map.len(), size - i - 1);
        }

        for i in 0..size / 2 {
            assert_eq!(map.get(&(2 * i)), None);
            assert_eq!(map.get(&(2 * i + 1)).unwrap(), &(i * 200 + 100));
        }

        for i in 0..size / 2 {
            assert_eq!(map.remove(&(2 * i)), None);
            assert_eq!(map.remove(&(2 * i + 1)), Some(i * 200 + 100));
            assert_eq!(map.len(), size / 2 - i - 1);
        }
    }

    #[test]
    fn test_basic_small() {
        let mut map = SortedVecMap::new();
        assert_eq!(map.remove(&1), None);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.insert(1, 1), None);
        assert_eq!(map.get(&1), Some(&1));
        assert_eq!(map.insert(1, 2), Some(1));
        assert_eq!(map.get(&1), Some(&2));
        assert_eq!(map.insert(2, 4), None);
        assert_eq!(map.get(&2), Some(&4));
        assert_eq!(map.remove(&1), Some(2));
        assert_eq!(map.remove(&2), Some(4));
        assert_eq!(map.remove(&1), None);
    }

    #[test]
    fn test_iter() {
        #[cfg(not(miri))] // Miri is too slow
        let size = 10000;
        #[cfg(miri)]
        let size = 200;

        // Forwards
        let mut map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = (usize, usize)>,
        {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }
        test(size, map.iter().map(|(&k, &v)| (k, v)));
        test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
        test(size, map.into_iter());
    }

    #[test]
    fn test_iter_rev() {
        #[cfg(not(miri))] // Miri is too slow
        let size = 10000;
        #[cfg(miri)]
        let size = 200;

        // Forwards
        let mut map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = (usize, usize)>,
        {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (size - i - 1, size - i - 1));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }
        test(size, map.iter().rev().map(|(&k, &v)| (k, v)));
        test(size, map.iter_mut().rev().map(|(&k, &mut v)| (k, v)));
        test(size, map.into_iter().rev());
    }

    #[test]
    fn test_values_mut() {
        let mut a = SortedVecMap::new();
        a.insert(1, String::from("hello"));
        a.insert(2, String::from("goodbye"));

        for value in a.values_mut() {
            value.push_str("!");
        }

        let values: Vec<String> = a.values().cloned().collect();
        assert_eq!(values, [String::from("hello!"), String::from("goodbye!")]);
    }

    #[test]
    fn test_iter_mixed() {
        #[cfg(not(miri))] // Miri is too slow
        let size = 10000;
        #[cfg(miri)]
        let size = 200;

        // Forwards
        let mut map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test<T>(size: usize, mut iter: T)
        where
            T: Iterator<Item = (usize, usize)> + DoubleEndedIterator,
        {
            for i in 0..size / 4 {
                assert_eq!(iter.size_hint(), (size - i * 2, Some(size - i * 2)));
                assert_eq!(iter.next().unwrap(), (i, i));
                assert_eq!(iter.next_back().unwrap(), (size - i - 1, size - i - 1));
            }
            for i in size / 4..size * 3 / 4 {
                assert_eq!(iter.size_hint(), (size * 3 / 4 - i, Some(size * 3 / 4 - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }
        test(size, map.iter().map(|(&k, &v)| (k, v)));
        test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
        test(size, map.into_iter());
    }

    #[test]
    fn test_range_small() {
        let size = 5;

        // Forwards
        let map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        let mut j = 0;
        for ((&k, &v), i) in map.range(2..).zip(2..size) {
            assert_eq!(k, i);
            assert_eq!(v, i);
            j += 1;
        }
        assert_eq!(j, size - 2);
    }

    #[test]
    fn test_range_inclusive() {
        let size = 500;

        let map: SortedVecMap<_, _> = (0..=size).map(|i| (i, i)).collect();

        fn check<'a, L, R>(lhs: L, rhs: R)
        where
            L: IntoIterator<Item = (&'a i32, &'a i32)>,
            R: IntoIterator<Item = (&'a i32, &'a i32)>,
        {
            let lhs: Vec<_> = lhs.into_iter().collect();
            let rhs: Vec<_> = rhs.into_iter().collect();
            assert_eq!(lhs, rhs);
        }

        check(map.range(size + 1..=size + 1), vec![]);
        check(map.range(size..=size), vec![(&size, &size)]);
        check(map.range(size..=size + 1), vec![(&size, &size)]);
        check(map.range(0..=0), vec![(&0, &0)]);
        check(map.range(0..=size - 1), map.range(..size));
        check(map.range(-1..=-1), vec![]);
        check(map.range(-1..=size), map.range(..));
        check(map.range(..=size), map.range(..));
        check(map.range(..=200), map.range(..201));
        check(
            map.range(5..=8),
            vec![(&5, &5), (&6, &6), (&7, &7), (&8, &8)],
        );
        check(map.range(-1..=0), vec![(&0, &0)]);
        check(map.range(-1..=2), vec![(&0, &0), (&1, &1), (&2, &2)]);
    }

    #[test]
    fn test_range_inclusive_max_value() {
        let max = std::usize::MAX;
        let map: SortedVecMap<_, _> = vec![(max, 0)].into_iter().collect();

        assert_eq!(map.range(max..=max).collect::<Vec<_>>(), &[(&max, &0)]);
    }

    #[test]
    fn test_range_equal_empty_cases() {
        let map: SortedVecMap<_, _> = (0..5).map(|i| (i, i)).collect();
        assert_eq!(map.range((Included(2), Excluded(2))).next(), None);
        assert_eq!(map.range((Excluded(2), Included(2))).next(), None);
    }

    #[test]
    #[should_panic]
    #[cfg(not(miri))] // Miri does not support panics
    fn test_range_equal_excluded() {
        let map: SortedVecMap<_, _> = (0..5).map(|i| (i, i)).collect();
        map.range((Excluded(2), Excluded(2)));
    }

    #[test]
    #[should_panic]
    #[cfg(not(miri))] // Miri does not support panics
    fn test_range_backwards_1() {
        let map: SortedVecMap<_, _> = (0..5).map(|i| (i, i)).collect();
        map.range((Included(3), Included(2)));
    }

    #[test]
    #[should_panic]
    #[cfg(not(miri))] // Miri does not support panics
    fn test_range_backwards_2() {
        let map: SortedVecMap<_, _> = (0..5).map(|i| (i, i)).collect();
        map.range((Included(3), Excluded(2)));
    }

    #[test]
    #[should_panic]
    #[cfg(not(miri))] // Miri does not support panics
    fn test_range_backwards_3() {
        let map: SortedVecMap<_, _> = (0..5).map(|i| (i, i)).collect();
        map.range((Excluded(3), Included(2)));
    }

    #[test]
    #[should_panic]
    #[cfg(not(miri))] // Miri does not support panics
    fn test_range_backwards_4() {
        let map: SortedVecMap<_, _> = (0..5).map(|i| (i, i)).collect();
        map.range((Excluded(3), Excluded(2)));
    }

    #[test]
    fn test_range_1000() {
        #[cfg(not(miri))] // Miri is too slow
        let size = 1000;
        #[cfg(miri)]
        let size = 200;
        let map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test(map: &SortedVecMap<u32, u32>, size: u32, min: Bound<&u32>, max: Bound<&u32>) {
            let mut kvs = map.range((min, max)).map(|(&k, &v)| (k, v));
            let mut pairs = (0..size).map(|i| (i, i));

            for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                assert_eq!(kv, pair);
            }
            assert_eq!(kvs.next(), None);
            assert_eq!(pairs.next(), None);
        }
        test(&map, size, Included(&0), Excluded(&size));
        test(&map, size, Unbounded, Excluded(&size));
        test(&map, size, Included(&0), Included(&(size - 1)));
        test(&map, size, Unbounded, Included(&(size - 1)));
        test(&map, size, Included(&0), Unbounded);
        test(&map, size, Unbounded, Unbounded);
    }

    #[test]
    fn test_range_borrowed_key() {
        let mut map = SortedVecMap::new();
        map.insert("aardvark".to_string(), 1);
        map.insert("baboon".to_string(), 2);
        map.insert("coyote".to_string(), 3);
        map.insert("dingo".to_string(), 4);
        // NOTE: would like to use simply "b".."d" here...
        let mut iter = map.range::<str, _>((Included("b"), Excluded("d")));
        assert_eq!(iter.next(), Some((&"baboon".to_string(), &2)));
        assert_eq!(iter.next(), Some((&"coyote".to_string(), &3)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_range() {
        #[cfg(not(miri))] // Miri is too slow
        let size = 200;
        #[cfg(miri)]
        let size = 30;
        let map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        for i in 0..size {
            for j in i..size {
                let mut kvs = map
                    .range((Included(&i), Included(&j)))
                    .map(|(&k, &v)| (k, v));
                let mut pairs = (i..=j).map(|i| (i, i));

                for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                    assert_eq!(kv, pair);
                }
                assert_eq!(kvs.next(), None);
                assert_eq!(pairs.next(), None);
            }
        }
    }

    #[test]
    fn test_range_mut() {
        #[cfg(not(miri))] // Miri is too slow
        let size = 200;
        #[cfg(miri)]
        let size = 30;
        let mut map: SortedVecMap<_, _> = (0..size).map(|i| (i, i)).collect();

        for i in 0..size {
            for j in i..size {
                let mut kvs = map
                    .range_mut((Included(&i), Included(&j)))
                    .map(|(&k, &mut v)| (k, v));
                let mut pairs = (i..=j).map(|i| (i, i));

                for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                    assert_eq!(kv, pair);
                }
                assert_eq!(kvs.next(), None);
                assert_eq!(pairs.next(), None);
            }
        }
    }

    #[test]
    fn test_borrow() {
        // make sure these compile -- using the Borrow trait
        {
            let mut map = SortedVecMap::new();
            map.insert("0".to_string(), 1);
            assert_eq!(map["0"], 1);
        }

        {
            let mut map = SortedVecMap::new();
            map.insert(Box::new(0), 1);
            assert_eq!(map[&0], 1);
        }

        {
            let mut map = SortedVecMap::new();
            map.insert(Box::new([0, 1]) as Box<[i32]>, 1);
            assert_eq!(map[&[0, 1][..]], 1);
        }

        {
            let mut map = SortedVecMap::new();
            map.insert(Rc::new(0), 1);
            assert_eq!(map[&0], 1);
        }
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: SortedVecMap<_, _> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        assert_eq!(map.get(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);

        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                *v *= 10;
            }
        }
        assert_eq!(map.get(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);

        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_extend_ref() {
        let mut a = SortedVecMap::new();
        a.insert(1, "one");
        let mut b = SortedVecMap::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[&1], "one");
        assert_eq!(a[&2], "two");
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn test_zst() {
        let mut m = SortedVecMap::new();
        assert_eq!(m.len(), 0);

        assert_eq!(m.insert((), ()), None);
        assert_eq!(m.len(), 1);

        assert_eq!(m.insert((), ()), Some(()));
        assert_eq!(m.len(), 1);
        assert_eq!(m.iter().count(), 1);

        m.clear();
        assert_eq!(m.len(), 0);

        for _ in 0..100 {
            m.insert((), ());
        }

        assert_eq!(m.len(), 1);
        assert_eq!(m.iter().count(), 1);
    }

    // This test's only purpose is to ensure that zero-sized keys with nonsensical orderings
    // do not cause segfaults when used with zero-sized values. All other map behavior is
    // undefined.
    #[test]
    fn test_bad_zst() {
        use std::cmp::Ordering;

        struct Bad;

        impl PartialEq for Bad {
            fn eq(&self, _: &Self) -> bool {
                false
            }
        }

        impl Eq for Bad {}

        impl PartialOrd for Bad {
            fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
                Some(Ordering::Less)
            }
        }

        impl Ord for Bad {
            fn cmp(&self, _: &Self) -> Ordering {
                Ordering::Less
            }
        }

        let mut m = SortedVecMap::new();

        for _ in 0..100 {
            m.insert(Bad, Bad);
        }
    }

    #[test]
    fn test_clone() {
        let mut map = SortedVecMap::new();
        #[cfg(not(miri))] // Miri is too slow
        let size = 100;
        #[cfg(miri)]
        let size = 30;
        assert_eq!(map.len(), 0);

        for i in 0..size {
            assert_eq!(map.insert(i, 10 * i), None);
            assert_eq!(map.len(), i + 1);
            assert_eq!(map, map.clone());
        }

        for i in 0..size {
            assert_eq!(map.insert(i, 100 * i), Some(10 * i));
            assert_eq!(map.len(), size);
            assert_eq!(map, map.clone());
        }

        for i in 0..size / 2 {
            assert_eq!(map.remove(&(i * 2)), Some(i * 200));
            assert_eq!(map.len(), size - i - 1);
            assert_eq!(map, map.clone());
        }

        for i in 0..size / 2 {
            assert_eq!(map.remove(&(2 * i)), None);
            assert_eq!(map.remove(&(2 * i + 1)), Some(i * 200 + 100));
            assert_eq!(map.len(), size / 2 - i - 1);
            assert_eq!(map, map.clone());
        }
    }

    #[test]
    #[allow(dead_code)]
    fn test_variance() {
        use std::collections::btree_map::{IntoIter, Iter, Keys, Range, Values};

        fn map_key<'new>(v: SortedVecMap<&'static str, ()>) -> SortedVecMap<&'new str, ()> {
            v
        }
        fn map_val<'new>(v: SortedVecMap<(), &'static str>) -> SortedVecMap<(), &'new str> {
            v
        }
        fn iter_key<'a, 'new>(v: Iter<'a, &'static str, ()>) -> Iter<'a, &'new str, ()> {
            v
        }
        fn iter_val<'a, 'new>(v: Iter<'a, (), &'static str>) -> Iter<'a, (), &'new str> {
            v
        }
        fn into_iter_key<'new>(v: IntoIter<&'static str, ()>) -> IntoIter<&'new str, ()> {
            v
        }
        fn into_iter_val<'new>(v: IntoIter<(), &'static str>) -> IntoIter<(), &'new str> {
            v
        }
        fn range_key<'a, 'new>(v: Range<'a, &'static str, ()>) -> Range<'a, &'new str, ()> {
            v
        }
        fn range_val<'a, 'new>(v: Range<'a, (), &'static str>) -> Range<'a, (), &'new str> {
            v
        }
        fn keys<'a, 'new>(v: Keys<'a, &'static str, ()>) -> Keys<'a, &'new str, ()> {
            v
        }
        fn vals<'a, 'new>(v: Values<'a, (), &'static str>) -> Values<'a, (), &'new str> {
            v
        }
    }

    #[test]
    fn test_occupied_entry_key() {
        let mut a = SortedVecMap::new();
        let key = "hello there";
        let value = "value goes here";
        assert!(a.is_empty());
        a.insert(key.clone(), value.clone());
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);

        match a.entry(key.clone()) {
            Vacant(_) => panic!(),
            Occupied(e) => assert_eq!(key, *e.key()),
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_vacant_entry_key() {
        let mut a = SortedVecMap::new();
        let key = "hello there";
        let value = "value goes here";

        assert!(a.is_empty());
        match a.entry(key.clone()) {
            Occupied(_) => panic!(),
            Vacant(e) => {
                assert_eq!(key, *e.key());
                e.insert(value.clone());
            }
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    fn rand_data(len: usize) -> Vec<(u32, u32)> {
        let mut rng = DeterministicRng::new();
        Vec::from_iter((0..len).map(|_| (rng.next(), rng.next())))
    }

    #[test]
    fn test_split_off_empty_right() {
        let mut data = rand_data(173);

        let mut map = SortedVecMap::from_iter(data.clone());
        let right = map.split_off(&(data.iter().max().unwrap().0 + 1));

        data.sort();
        assert!(map.into_iter().eq(data));
        assert!(right.into_iter().eq(None));
    }

    #[test]
    fn test_split_off_empty_left() {
        let mut data = rand_data(314);

        let mut map = SortedVecMap::from_iter(data.clone());
        let right = map.split_off(&data.iter().min().unwrap().0);

        data.sort();
        assert!(map.into_iter().eq(None));
        assert!(right.into_iter().eq(data));
    }

    #[test]
    fn test_split_off_large_random_sorted() {
        let mut data = rand_data(1529);
        // special case with maximum height.
        data.sort();

        let mut map = SortedVecMap::from_iter(data.clone());
        let key = data[data.len() / 2].0;
        let right = map.split_off(&key);

        assert!(map
            .into_iter()
            .eq(data.clone().into_iter().filter(|x| x.0 < key)));
        assert!(right
            .into_iter()
            .eq(data.into_iter().filter(|x| x.0 >= key)));
    }

    #[test]

    fn test_serialization() {
        let mut map: SortedVecMap<i64, u32> = SortedVecMap::new();
        map.insert(-5, 10);
        map.insert(12, 0);
        map.insert(55, 1213);
        map.insert(-49, 5);
        map.insert(10, 2);

        let map2: SortedVecMap<i64, u32> =
            serde_json::from_str(&serde_json::to_string(&map).unwrap()).unwrap();

        assert_eq!(map2, map);
    }

    #[test]
    fn test_macro() {
        let map = sorted_vec_map! {
            60 => 12,
            10 => 0,
            222 => 54,
            44 => 22,
            44 => 10
        };

        let mut map2 = SortedVecMap::new();
        map2.insert(60, 12);
        map2.insert(10, 0);
        map2.insert(222, 54);
        map2.insert(44, 10);
        map2.insert(44, 22);

        assert_eq!(map, map2);
    }
}
