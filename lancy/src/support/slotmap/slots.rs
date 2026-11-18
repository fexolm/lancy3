use super::{DenseMap, SlotKey};
use std::ops::{Index, IndexMut};

/// Stable keys for entities that can be removed. Removed slots are never reused.
///
/// ```compile_fail
/// use lancy::{slot_key, support::slotmap::SlotMap};
/// slot_key!(Block);
/// slot_key!(Inst);
/// let mut blocks = SlotMap::<Block, ()>::new();
/// let mut insts = SlotMap::<Inst, ()>::new();
/// let inst = insts.insert(());
/// blocks.remove(inst);
/// ```
#[derive(Clone, Debug)]
pub struct SlotMap<K, V> {
    slots: DenseMap<K, Option<V>>,
    live: usize,
}

impl<K: SlotKey, V> SlotMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, value: V) -> K {
        let key = self.slots.insert(Some(value));
        self.live += 1;
        key
    }

    pub fn remove(&mut self, key: K) -> Option<V> {
        let value = self.slots.get_mut(key)?.take()?;
        self.live -= 1;
        Some(value)
    }

    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Number of entries a dense side table needs to cover every allocated key.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn get(&self, key: K) -> Option<&V> {
        self.slots.get(key)?.as_ref()
    }
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.slots.get_mut(key)?.as_mut()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (K, &V)> {
        self.slots
            .iter()
            .filter_map(|(key, slot)| Some((key, slot.as_ref()?)))
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = K> {
        self.iter().map(|(key, _)| key)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.slots.values().flatten()
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut V> {
        self.slots.values_mut().flatten()
    }
}

impl<K, V> Default for SlotMap<K, V> {
    fn default() -> Self {
        Self {
            slots: DenseMap::default(),
            live: 0,
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
