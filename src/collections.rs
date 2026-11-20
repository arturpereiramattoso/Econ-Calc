use std::borrow::Borrow;

/// Deterministic insertion-ordered map (`IndexMap<K, V>`) preserving declaration sequence.
#[derive(Debug, Clone)]
pub struct IndexMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> Default for IndexMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for IndexMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

pub struct Iter<'a, K, V> {
    inner: std::slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}

pub struct IterMut<'a, K, V> {
    inner: std::slice::IterMut<'a, (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (&*k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}

pub struct Keys<'a, K, V> {
    inner: std::slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {}

pub struct Values<'a, K, V> {
    inner: std::slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {}

impl<K, V> IndexMap<K, V>
where
    K: PartialEq,
{
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        for (existing_k, existing_v) in &mut self.entries {
            if *existing_k == key {
                return Some(std::mem::replace(existing_v, value));
            }
        }
        self.entries.push((key, value));
        None
    }

    pub fn push_entry(&mut self, key: K, value: V) {
        self.entries.push((key, value));
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: PartialEq + ?Sized,
    {
        self.entries
            .iter()
            .find(|(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: PartialEq + ?Sized,
    {
        self.entries
            .iter_mut()
            .find(|(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn get_index(&self, idx: usize) -> Option<(&K, &V)> {
        self.entries.get(idx).map(|(k, v)| (k, v))
    }

    pub fn get_index_mut(&mut self, idx: usize) -> Option<(&K, &mut V)> {
        self.entries.get_mut(idx).map(|(k, v)| (&*k, v))
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: PartialEq + ?Sized,
    {
        self.entries.iter().any(|(k, _)| k.borrow() == key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: PartialEq + ?Sized,
    {
        if let Some(idx) = self.entries.iter().position(|(k, _)| k.borrow() == key) {
            Some(self.entries.remove(idx).1)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.entries.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.entries.iter_mut(),
        }
    }

    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys {
            inner: self.entries.iter(),
        }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        Values {
            inner: self.entries.iter(),
        }
    }

    pub fn as_slice(&self) -> &[(K, V)] {
        &self.entries
    }
}

impl<'a, K, V> IntoIterator for &'a IndexMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            inner: self.entries.iter(),
        }
    }
}

impl<'a, K, V> IntoIterator for &'a mut IndexMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            inner: self.entries.iter_mut(),
        }
    }
}
