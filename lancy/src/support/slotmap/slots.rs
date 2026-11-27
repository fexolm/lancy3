use super::{DenseMap, SlotKey};
use std::ops::{Index, IndexMut};

/// Stable keys for entities that can be removed. Removed slots are never reused.
#[derive(Clone, Debug)]
pub(crate) struct SlotMap<K, V> {
    slots: DenseMap<K, Option<V>>,
}

impl<K: SlotKey, V> SlotMap<K, V> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(&mut self, value: V) -> K {
        self.slots.insert(Some(value))
    }

    pub(crate) fn remove(&mut self, key: K) -> Option<V> {
        self.slots.get_mut(key)?.take()
    }

    /// Number of entries a dense side table needs to cover every allocated key.
    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn get(&self, key: K) -> Option<&V> {
        self.slots.get(key)?.as_ref()
    }
    pub(crate) fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.slots.get_mut(key)?.as_mut()
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (K, &V)> {
        self.slots
            .iter()
            .filter_map(|(key, slot)| Some((key, slot.as_ref()?)))
    }

    pub(crate) fn keys(&self) -> impl DoubleEndedIterator<Item = K> {
        self.iter().map(|(key, _)| key)
    }

    pub(crate) fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.slots.values().flatten()
    }

    pub(crate) fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut V> {
        self.slots.values_mut().flatten()
    }
}

impl<K, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self {
            slots: DenseMap::default(),
        }
    }
}

impl<K: SlotKey, V> Index<K> for SlotMap<K, V> {
    type Output = V;
    fn index(&self, key: K) -> &V {
        self.get(key).expect("missing slot")
    }
}

impl<K: SlotKey, V> IndexMut<K> for SlotMap<K, V> {
    fn index_mut(&mut self, key: K) -> &mut V {
        self.get_mut(key).expect("missing slot")
    }
}
