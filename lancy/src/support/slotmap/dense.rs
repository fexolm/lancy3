use super::SlotKey;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Contiguous storage for append-only entities or a side table in the same key space.
#[derive(Clone, Debug)]
pub(crate) struct DenseMap<K, V> {
    entries: Vec<V>,
    key: PhantomData<fn(K) -> K>,
}

impl<K: SlotKey, V> DenseMap<K, V> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            key: PhantomData,
        }
    }

    pub(crate) fn filled(len: usize, value: V) -> Self
    where
        V: Clone,
    {
        assert!(len <= u32::MAX as usize, "slot count exceeds 32-bit range");
        Self {
            entries: vec![value; len],
            key: PhantomData,
        }
    }

    pub(crate) fn from_fn(len: usize, mut make: impl FnMut(K) -> V) -> Self {
        assert!(len <= u32::MAX as usize, "slot count exceeds 32-bit range");
        let entries = (0..len).map(|i| make(K::from_index(i))).collect();
        Self {
            entries,
            key: PhantomData,
        }
    }

    pub(crate) fn insert(&mut self, value: V) -> K {
        let key = K::from_index(self.len());
        self.entries.push(value);
        key
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn get(&self, key: K) -> Option<&V> {
        self.entries.get(key.index())
    }
    pub(crate) fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.entries.get_mut(key.index())
    }

    pub(crate) fn keys(
        &self,
    ) -> impl DoubleEndedIterator<Item = K> + ExactSizeIterator + use<K, V> {
        (0..self.len()).map(K::from_index)
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (K, &V)> + ExactSizeIterator {
        self.keys().zip(&self.entries)
    }

    pub(crate) fn iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = (K, &mut V)> + ExactSizeIterator {
        self.keys().zip(&mut self.entries)
    }

    pub(crate) fn values(&self) -> std::slice::Iter<'_, V> {
        self.entries.iter()
    }
    pub(crate) fn values_mut(&mut self) -> std::slice::IterMut<'_, V> {
        self.entries.iter_mut()
    }
    pub(crate) fn as_slice(&self) -> &[V] {
        &self.entries
    }
}

impl<K, V> Default for DenseMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            key: PhantomData,
        }
    }
}

impl<K: SlotKey, V> Index<K> for DenseMap<K, V> {
    type Output = V;
    fn index(&self, key: K) -> &V {
        &self.entries[key.index()]
    }
}

impl<K: SlotKey, V> IndexMut<K> for DenseMap<K, V> {
    fn index_mut(&mut self, key: K) -> &mut V {
        &mut self.entries[key.index()]
    }
}
