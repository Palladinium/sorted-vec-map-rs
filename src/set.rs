use std::{
    borrow::Borrow,
    cmp::{
        max, min,
        Ordering::{self, *},
    },
    fmt::{self, Debug},
    hash::{self, Hash},
    iter::*,
    marker::PhantomData,
    mem,
    ops::RangeBounds,
    slice, vec,
};

use serde::{
    de::{Deserialize, Deserializer, SeqAccess, Visitor},
    ser::{Serialize, SerializeSeq, Serializer},
};

/// A structure similar to `BTreeSet`, implemented in terms of a sorted `Vec<K, V>`.
/// The API of this struct will attempt to be, by convention, as compatible to `BTreeSet` as possible.
pub struct SortedVecSet<T> {
    vec: Vec<T>,
}

impl<T: Debug + Ord> Debug for SortedVecSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Ord> Default for SortedVecSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq> PartialEq for SortedVecSet<T> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.vec.eq(&rhs.vec)
    }
}

impl<T: Eq> Eq for SortedVecSet<T> {}

impl<T: PartialOrd> PartialOrd for SortedVecSet<T> {
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.vec.partial_cmp(&rhs.vec)
    }
}

impl<T: Ord> Ord for SortedVecSet<T> {
    #[inline]
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.vec.cmp(&rhs.vec)
    }
}

impl<T: Ord> SortedVecSet<T> {
    // Replicated `Vec<K, V>` APIs follow
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
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        vec.sort_unstable();
        vec.dedup();

        Self { vec }
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
    pub fn into_vec(self) -> Vec<T> {
        self.vec
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.vec.truncate(len)
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }

    #[inline]
    pub fn remove_at(&mut self, index: usize) -> T {
        self.vec.remove(index)
    }

    #[inline]
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.vec.retain(f)
    }

    #[inline]
    pub fn drain<R>(&mut self, range: R) -> Drain<T>
    where
        R: RangeBounds<usize>,
    {
        Drain {
            iter: self.vec.drain(range),
        }
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
    pub fn split_off_at(&mut self, at: usize) -> SortedVecSet<T> {
        Self {
            vec: self.vec.split_off(at),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.vec.iter(),
        }
    }

    // Replicated `BTreeSet<T>` APIs follow

    #[inline]
    pub fn difference<'a>(&'a self, other: &'a SortedVecSet<T>) -> Difference<'a, T> {
        Difference {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    #[inline]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a SortedVecSet<T>,
    ) -> SymmetricDifference<'a, T> {
        SymmetricDifference {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a SortedVecSet<T>) -> Intersection<'a, T> {
        Intersection {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    #[inline]
    pub fn union<'a>(&'a self, other: &'a SortedVecSet<T>) -> Union<'a, T> {
        Union {
            a: self.iter().peekable(),
            b: other.iter().peekable(),
        }
    }

    #[inline]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
    {
        self.vec
            .binary_search_by(|item| item.borrow().cmp(value))
            .is_ok()
    }

    #[inline]
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
    {
        self.vec
            .binary_search_by(|item| item.borrow().cmp(value))
            .ok()
            .map(|idx| &self.vec[idx])
    }

    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).next().is_none()
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        let mut x = self.iter();
        let mut y = other.iter();
        let mut a = x.next();
        let mut b = y.next();
        while a.is_some() {
            if b.is_none() {
                return false;
            }

            let a1 = a.unwrap();
            let b1 = b.unwrap();

            match b1.cmp(a1) {
                Less => (),
                Greater => return false,
                Equal => a = x.next(),
            }

            b = y.next();
        }
        true
    }

    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        match self.vec.binary_search_by(|item| item.borrow().cmp(&value)) {
            Ok(_) => false,
            Err(idx) => {
                self.vec.insert(idx, value);
                true
            }
        }
    }

    #[inline]
    pub fn replace(&mut self, mut value: T) -> Option<T> {
        match self.vec.binary_search_by(|item| item.borrow().cmp(&value)) {
            Ok(idx) => {
                mem::swap(&mut self.vec[idx], &mut value);
                Some(value)
            }
            Err(idx) => {
                self.vec.insert(idx, value);
                None
            }
        }
    }

    #[inline]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
    {
        match self.vec.binary_search_by(|item| item.borrow().cmp(&value)) {
            Ok(idx) => {
                self.vec.remove(idx);
                true
            }
            Err(_) => false,
        }
    }

    #[inline]
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
    {
        self.vec
            .binary_search_by(|item| item.borrow().cmp(&value))
            .ok()
            .map(|idx| self.vec.remove(idx))
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        for item in other.vec.drain(..) {
            self.insert(item);
        }
    }

    #[inline]
    pub fn split_off<Q>(&mut self, key: &Q) -> SortedVecSet<T>
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
    {
        let idx = match self.vec.binary_search_by(|item| item.borrow().cmp(key)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        Self {
            vec: self.vec.split_off(idx),
        }
    }

    #[inline]
    pub fn position<Q>(&mut self, key: &Q) -> Result<usize, usize>
    where
        Q: Ord + ?Sized,
        T: Borrow<Q>,
    {
        self.vec.binary_search_by(|item| item.borrow().cmp(key))
    }
}

fn cmp_opt<T: Ord>(x: Option<&T>, y: Option<&T>, short: Ordering, long: Ordering) -> Ordering {
    match (x, y) {
        (None, _) => short,
        (_, None) => long,
        (Some(x1), Some(y1)) => x1.cmp(y1),
    }
}

pub struct Drain<'a, T> {
    iter: vec::Drain<'a, T>,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, T> FusedIterator for Drain<'a, T> {}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

#[derive(Debug, Clone)]
pub struct Difference<'a, T> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for Difference<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Less, Less) {
                Less => return self.a.next(),
                Equal => {
                    self.a.next();
                    self.b.next();
                }
                Greater => {
                    self.b.next();
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let a_len = self.a.len();
        let b_len = self.b.len();
        (a_len.saturating_sub(b_len), Some(a_len))
    }
}

impl<'a, T: Ord> FusedIterator for Difference<'a, T> {}

#[derive(Debug, Clone)]
pub struct SymmetricDifference<'a, T> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                Less => return self.a.next(),
                Equal => {
                    self.a.next();
                    self.b.next();
                }
                Greater => return self.b.next(),
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.a.len() + self.b.len()))
    }
}

impl<'a, T: Ord> FusedIterator for SymmetricDifference<'a, T> {}

#[derive(Debug, Clone)]
pub struct Intersection<'a, T> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for Intersection<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match Ord::cmp(self.a.peek()?, self.b.peek()?) {
                Less => {
                    self.a.next();
                }
                Equal => {
                    self.b.next();
                    return self.a.next();
                }
                Greater => {
                    self.b.next();
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(min(self.a.len(), self.b.len())))
    }
}

impl<'a, T: Ord> FusedIterator for Intersection<'a, T> {}

#[derive(Debug, Clone)]
pub struct Union<'a, T> {
    a: Peekable<Iter<'a, T>>,
    b: Peekable<Iter<'a, T>>,
}

impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
            Less => self.a.next(),
            Equal => {
                self.b.next();
                self.a.next()
            }
            Greater => self.b.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let a_len = self.a.len();
        let b_len = self.b.len();
        (max(a_len, b_len), Some(a_len + b_len))
    }
}

impl<'a, T: Ord> FusedIterator for Union<'a, T> {}

impl<T: Ord + Clone> SortedVecSet<T> {
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        for item in other {
            self.insert(item.clone());
        }
    }
}

impl<T: Ord> FromIterator<T> for SortedVecSet<T> {
    #[inline]
    fn from_iter<Iter>(iter: Iter) -> Self
    where
        Iter: IntoIterator<Item = T>,
    {
        Self::from_vec(Vec::from_iter(iter))
    }
}

impl<T: Ord + Clone> Clone for SortedVecSet<T> {
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

impl<T: Ord + Hash> Hash for SortedVecSet<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.vec.hash(state)
    }
}

impl<T> IntoIterator for SortedVecSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.vec.into_iter(),
        }
    }
}

#[derive(Debug)]
pub struct IntoIter<T> {
    iter: vec::IntoIter<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> FusedIterator for IntoIter<T> {}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, T: Ord> IntoIterator for &'a SortedVecSet<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct Iter<'a, T> {
    iter: slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T: Ord> Extend<T> for SortedVecSet<T> {
    #[inline]
    fn extend<Iter: IntoIterator<Item = T>>(&mut self, iter: Iter) {
        for elem in iter {
            self.insert(elem);
        }
    }
}

impl<'a, T: 'a + Ord + Copy> Extend<&'a T> for SortedVecSet<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

impl<T: Ord> Into<Vec<T>> for SortedVecSet<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.into_vec()
    }
}

impl<T: Ord + Serialize> Serialize for SortedVecSet<T> where {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;

        for e in self.iter() {
            seq.serialize_element(e)?;
        }

        seq.end()
    }
}

impl<'de, T: Ord + Deserialize<'de>> Deserialize<'de> for SortedVecSet<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(SortedVecSetVisitor::new())
    }
}

struct SortedVecSetVisitor<T>(PhantomData<T>);

impl<T> SortedVecSetVisitor<T> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'de, T: Ord + Deserialize<'de>> Visitor<'de> for SortedVecSetVisitor<T> {
    type Value = SortedVecSet<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut set = SortedVecSet::with_capacity(access.size_hint().unwrap_or(0));

        // While there are entries remaining in the input, add them
        // into our map.
        while let Some(element) = access.next_element()? {
            set.insert(element);
        }

        Ok(set)
    }
}

#[macro_export]
macro_rules! sorted_vec_set {
    ($($e:expr),*) => { SortedVecSet::from_vec(vec![$($e),*])};
}

#[cfg(test)]
mod tests {
    use super::{super::test_utils::DeterministicRng, *};

    use std::{collections::hash_map::DefaultHasher, hash::Hasher};

    #[test]
    fn test_clone_eq() {
        let mut m = SortedVecSet::new();

        m.insert(1);
        m.insert(2);

        assert!(m.clone() == m);
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    #[test]
    fn test_hash() {
        let mut x = SortedVecSet::new();
        let mut y = SortedVecSet::new();

        x.insert(1);
        x.insert(2);
        x.insert(3);

        y.insert(3);
        y.insert(2);
        y.insert(1);

        assert!(hash(&x) == hash(&y));
    }

    fn check<F>(a: &[i32], b: &[i32], expected: &[i32], f: F)
    where
        F: FnOnce(&SortedVecSet<i32>, &SortedVecSet<i32>, &mut dyn FnMut(&i32) -> bool) -> bool,
    {
        let mut set_a = SortedVecSet::new();
        let mut set_b = SortedVecSet::new();

        for x in a {
            assert!(set_a.insert(*x))
        }
        for y in b {
            assert!(set_b.insert(*y))
        }

        let mut i = 0;
        f(&set_a, &set_b, &mut |&x| {
            assert_eq!(x, expected[i]);
            i += 1;
            true
        });
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_intersection() {
        fn check_intersection(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.intersection(y).all(f))
        }

        check_intersection(&[], &[], &[]);
        check_intersection(&[1, 2, 3], &[], &[]);
        check_intersection(&[], &[1, 2, 3], &[]);
        check_intersection(&[2], &[1, 2, 3], &[2]);
        check_intersection(&[1, 2, 3], &[2], &[2]);
        check_intersection(
            &[11, 1, 3, 77, 103, 5, -5],
            &[2, 11, 77, -9, -42, 5, 3],
            &[3, 5, 11, 77],
        );
    }

    #[test]
    fn test_difference() {
        fn check_difference(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.difference(y).all(f))
        }

        check_difference(&[], &[], &[]);
        check_difference(&[1, 12], &[], &[1, 12]);
        check_difference(&[], &[1, 2, 3, 9], &[]);
        check_difference(&[1, 3, 5, 9, 11], &[3, 9], &[1, 5, 11]);
        check_difference(
            &[-5, 11, 22, 33, 40, 42],
            &[-12, -5, 14, 23, 34, 38, 39, 50],
            &[11, 22, 33, 40, 42],
        );
    }

    #[test]
    fn test_symmetric_difference() {
        fn check_symmetric_difference(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.symmetric_difference(y).all(f))
        }

        check_symmetric_difference(&[], &[], &[]);
        check_symmetric_difference(&[1, 2, 3], &[2], &[1, 3]);
        check_symmetric_difference(&[2], &[1, 2, 3], &[1, 3]);
        check_symmetric_difference(
            &[1, 3, 5, 9, 11],
            &[-2, 3, 9, 14, 22],
            &[-2, 1, 5, 11, 14, 22],
        );
    }

    #[test]
    fn test_union() {
        fn check_union(a: &[i32], b: &[i32], expected: &[i32]) {
            check(a, b, expected, |x, y, f| x.union(y).all(f))
        }

        check_union(&[], &[], &[]);
        check_union(&[1, 2, 3], &[2], &[1, 2, 3]);
        check_union(&[2], &[1, 2, 3], &[1, 2, 3]);
        check_union(
            &[1, 3, 5, 9, 11, 16, 19, 24],
            &[-2, 1, 5, 9, 13, 19],
            &[-2, 1, 3, 5, 9, 11, 13, 16, 19, 24],
        );
    }

    #[test]
    fn test_zip() {
        let mut x = SortedVecSet::new();
        x.insert(5);
        x.insert(12);
        x.insert(11);

        let mut y = SortedVecSet::new();
        y.insert("foo");
        y.insert("bar");

        let x = x;
        let y = y;
        let mut z = x.iter().zip(&y);

        assert_eq!(z.next().unwrap(), (&5, &("bar")));
        assert_eq!(z.next().unwrap(), (&11, &("foo")));
        assert!(z.next().is_none());
    }

    #[test]
    fn test_from_iter() {
        let xs = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: SortedVecSet<_> = xs.iter().cloned().collect();

        for x in &xs {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_show() {
        let mut set = SortedVecSet::new();
        let empty = SortedVecSet::<i32>::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{:?}", set);

        assert_eq!(set_str, "{1, 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_extend_ref() {
        let mut a = SortedVecSet::new();
        a.insert(1);

        a.extend(&[2, 3, 4]);

        assert_eq!(a.len(), 4);
        assert!(a.contains(&1));
        assert!(a.contains(&2));
        assert!(a.contains(&3));
        assert!(a.contains(&4));

        let mut b = SortedVecSet::new();
        b.insert(5);
        b.insert(6);

        a.extend(&b);

        assert_eq!(a.len(), 6);
        assert!(a.contains(&1));
        assert!(a.contains(&2));
        assert!(a.contains(&3));
        assert!(a.contains(&4));
        assert!(a.contains(&5));
        assert!(a.contains(&6));
    }

    #[test]
    fn test_recovery() {
        use std::cmp::Ordering;

        #[derive(Debug)]
        struct Foo(&'static str, i32);

        impl PartialEq for Foo {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl Eq for Foo {}

        impl PartialOrd for Foo {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl Ord for Foo {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.cmp(&other.0)
            }
        }

        let mut s = SortedVecSet::new();
        assert_eq!(s.replace(Foo("a", 1)), None);
        assert_eq!(s.len(), 1);
        assert_eq!(s.replace(Foo("a", 2)), Some(Foo("a", 1)));
        assert_eq!(s.len(), 1);

        {
            let mut it = s.iter();
            assert_eq!(it.next(), Some(&Foo("a", 2)));
            assert_eq!(it.next(), None);
        }

        assert_eq!(s.get(&Foo("a", 1)), Some(&Foo("a", 2)));
        assert_eq!(s.take(&Foo("a", 1)), Some(Foo("a", 2)));
        assert_eq!(s.len(), 0);

        assert_eq!(s.get(&Foo("a", 1)), None);
        assert_eq!(s.take(&Foo("a", 1)), None);

        assert_eq!(s.iter().next(), None);
    }

    #[test]
    #[allow(dead_code)]
    fn test_variance() {
        use std::collections::btree_set::{IntoIter, Iter, Range};

        fn set<'new>(v: SortedVecSet<&'static str>) -> SortedVecSet<&'new str> {
            v
        }
        fn iter<'a, 'new>(v: Iter<'a, &'static str>) -> Iter<'a, &'new str> {
            v
        }
        fn into_iter<'new>(v: IntoIter<&'static str>) -> IntoIter<&'new str> {
            v
        }
        fn range<'a, 'new>(v: Range<'a, &'static str>) -> Range<'a, &'new str> {
            v
        }
    }

    #[test]
    fn test_append() {
        let mut a = SortedVecSet::new();
        a.insert(1);
        a.insert(2);
        a.insert(3);

        let mut b = SortedVecSet::new();
        b.insert(3);
        b.insert(4);
        b.insert(5);

        a.append(&mut b);

        assert_eq!(a.len(), 5);
        assert_eq!(b.len(), 0);

        assert_eq!(a.contains(&1), true);
        assert_eq!(a.contains(&2), true);
        assert_eq!(a.contains(&3), true);
        assert_eq!(a.contains(&4), true);
        assert_eq!(a.contains(&5), true);
    }

    fn rand_data(len: usize) -> Vec<u32> {
        let mut rng = DeterministicRng::new();
        Vec::from_iter((0..len).map(|_| rng.next()))
    }

    #[test]
    fn test_split_off_empty_right() {
        let mut data = rand_data(173);

        let mut set = SortedVecSet::from_iter(data.clone());
        let right = set.split_off(&(data.iter().max().unwrap() + 1));

        data.sort();
        assert!(set.into_iter().eq(data));
        assert!(right.into_iter().eq(Option::<u32>::None));
    }

    #[test]
    fn test_split_off_empty_left() {
        let mut data = rand_data(314);

        let mut set = SortedVecSet::from_iter(data.clone());
        let right = set.split_off(data.iter().min().unwrap());

        data.sort();
        assert!(set.into_iter().eq(Option::<u32>::None));
        assert!(right.into_iter().eq(data));
    }

    #[test]
    fn test_split_off_large_random_sorted() {
        let mut data = rand_data(1529);
        // special case with maximum height.
        data.sort();

        let mut set = SortedVecSet::from_iter(data.clone());
        let key = data[data.len() / 2];
        let right = set.split_off(&key);

        assert!(set
            .into_iter()
            .eq(data.clone().into_iter().filter(|x| *x < key)));
        assert!(right.into_iter().eq(data.into_iter().filter(|x| *x >= key)));
    }

    #[test]
    fn test_macro() {
        let set = sorted_vec_set![60, 10, 222, 44, 44];

        let mut set2 = SortedVecSet::new();
        set2.insert(60);
        set2.insert(10);
        set2.insert(222);
        set2.insert(44);
        set2.insert(44);

        assert_eq!(set, set2);
    }
}
