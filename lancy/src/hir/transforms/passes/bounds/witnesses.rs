//! A query point sees the combined facts in its enclosing dominance scopes.
use std::{
    cmp::Reverse,
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
    ops::Range,
};

struct Scope<V> {
    interval: Range<usize>,
    value: V,
}

enum Scopes<V> {
    One(Scope<V>),
    Many(Vec<Scope<V>>),
}

pub(super) struct Witnesses<K, V>(HashMap<K, Scopes<V>>);

impl<K, V> Default for Witnesses<K, V> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<K: Eq + Hash, V: Copy> Witnesses<K, V> {
    pub(super) fn record(&mut self, key: K, scope: Range<usize>, value: V) {
        let scope = Scope {
            interval: scope,
            value,
        };
        match self.0.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(Scopes::One(scope));
            }
            Entry::Occupied(mut entry) => {
                let scopes = entry.get_mut();
                match scopes {
                    Scopes::Many(scopes) => scopes.push(scope),
                    Scopes::One(previous) => {
                        let previous = Scope {
                            interval: previous.interval.clone(),
                            value: previous.value,
                        };
                        *scopes = Scopes::Many(vec![previous, scope]);
                    }
                }
            }
        }
    }

    pub(super) fn index(mut self, combine: impl Fn(V, V) -> V) -> ScopedFacts<K, V> {
        let mut active: Vec<(usize, V)> = Vec::new();
        let mut indexed = Vec::new();
        for scopes in self.0.values_mut() {
            let Scopes::Many(scopes) = scopes else {
                continue;
            };
            scopes.sort_unstable_by_key(|s| (s.interval.start, Reverse(s.interval.end)));
            indexed.clear();
            let mut start = 0;
            let mut append = |end, value| {
                if start < end
                    && let Some(value) = value
                {
                    indexed.push(Scope {
                        interval: start..end,
                        value,
                    });
                }
                start = end;
            };
            for scope in scopes.iter() {
                // Dominance scopes are nested or disjoint. Closing a child
                // restores the combined value of the enclosing scope.
                while active
                    .last()
                    .is_some_and(|&(end, _)| end <= scope.interval.start)
                {
                    let (end, value) = active.pop().unwrap();
                    append(end, Some(value));
                }
                let previous = active.last().map(|&(_, value)| value);
                append(scope.interval.start, previous);
                let value = previous.map_or(scope.value, |old| combine(old, scope.value));
                active.push((scope.interval.end, value));
            }
            while let Some((end, value)) = active.pop() {
                append(end, Some(value));
            }
            std::mem::swap(scopes, &mut indexed);
        }
        ScopedFacts(self.0)
    }
}

pub(super) struct ScopedFacts<K, V>(HashMap<K, Scopes<V>>);

impl<K: Eq + Hash, V: Copy> ScopedFacts<K, V> {
    pub(super) fn at(&self, key: &K, position: usize) -> Option<V> {
        let scope = match self.0.get(key)? {
            Scopes::One(scope) => scope,
            Scopes::Many(scopes) => {
                let end = scopes.partition_point(|s| s.interval.start <= position);
                &scopes[end.checked_sub(1)?]
            }
        };
        scope.interval.contains(&position).then_some(scope.value)
    }
}
