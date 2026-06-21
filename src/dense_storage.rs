use std::{collections::VecDeque, sync::atomic::AtomicU32};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index {
    index: u32,
    generation: u32,
}

#[derive(Default)]
pub struct IndexAllocator {
    next_index: AtomicU32,
    // TODO: Use channel instead if mutable access will cause problems
    recycle_queue: VecDeque<Index>,
    recycled: Vec<Index>,
}

impl IndexAllocator {
    pub fn reserve(&mut self) -> Index {
        if let Some(mut recycled) = self.recycle_queue.pop_front() {
            recycled.generation += 1;
            self.recycled.push(recycled);
            recycled
        } else {
            Index {
                index: self
                    .next_index
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                generation: 0,
            }
        }
    }

    pub fn recycle(&mut self, index: Index) {
        self.recycle_queue.push_back(index);
    }
}

struct Entry<T> {
    value: Option<T>,
    generation: u32,
}

pub struct DenseStorage<T> {
    buffer: Vec<Entry<T>>,
    len: u32,
    index_allocator: IndexAllocator,
}

impl<T> Default for DenseStorage<T> {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            len: 0,
            index_allocator: IndexAllocator::default(),
        }
    }
}

impl<T> DenseStorage<T> {
    /// Returns the number of stored items.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns `true` if there's no items stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn iter_buffer_mut(&mut self) -> impl IntoIterator<Item = &mut T> {
        self.buffer
            .iter_mut()
            .filter_map(|entry| entry.value.as_mut())
    }

    pub fn iter_buffer(&self) -> impl IntoIterator<Item = &T> {
        self.buffer.iter().filter_map(|entry| entry.value.as_ref())
    }

    pub fn index_allocator_mut(&mut self) -> &mut IndexAllocator {
        &mut self.index_allocator
    }

    pub fn insert(&mut self, index: Index, value: T) -> Result<bool, InvalidGenerationError> {
        self.flush();
        let entry = &mut self.buffer[index.index as usize];
        if entry.generation == index.generation {
            let exists = entry.value.is_some();
            // If it didn't exists that means we're adding a new item.
            if !exists {
                self.len += 1;
            }
            entry.value = Some(value);
            Ok(exists)
        } else {
            Err(InvalidGenerationError {
                index,
                current_generation: entry.generation,
            })
        }
    }

    pub fn push(&mut self, value: T) -> Index {
        let index = self.index_allocator_mut().reserve();
        // This insertion can't fail because we use valid index
        let _ = self.insert(index, value);
        index
    }

    /// Remove item from storage and queues index to be recycled.
    pub fn remove_recycle(&mut self, index: Index) -> Option<T> {
        self.remove(index)
            .inspect(|_| self.index_allocator.recycle(index))
    }

    /// Removes item from storage.
    ///
    /// **This method does not queue the index to be recycled.**
    pub fn remove(&mut self, index: Index) -> Option<T> {
        self.flush();
        let entry = &mut self.buffer[index.index as usize];
        if entry.generation == index.generation {
            entry.value.take().inspect(|_| self.len -= 1)
        } else {
            None
        }
    }

    pub fn get(&self, index: Index) -> Option<&T> {
        let entry = self.buffer.get(index.index as usize)?;
        if entry.generation == index.generation {
            entry.value.as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: Index) -> Option<&mut T> {
        let entry = self.buffer.get_mut(index.index as usize)?;
        if entry.generation == index.generation {
            entry.value.as_mut()
        } else {
            None
        }
    }

    fn flush(&mut self) {
        let new_len = self
            .index_allocator
            .next_index
            .load(std::sync::atomic::Ordering::Relaxed);

        self.buffer.resize_with(new_len as usize, || Entry {
            value: None,
            generation: 0,
        });

        for index in self.index_allocator.recycled.drain(..) {
            let entry = &mut self.buffer[index.index as usize];
            *entry = Entry {
                value: None,
                generation: index.generation,
            }
        }
    }
}

#[derive(Error, Debug)]
#[error("{index:?} has invalid generation. Current generation is {current_generation}")]
pub struct InvalidGenerationError {
    index: Index,
    current_generation: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_test() {
        let mut storage = DenseStorage::<i32>::default();

        let a = storage.index_allocator_mut().reserve();
        assert!(!storage.insert(a, 1).unwrap());

        let b = storage.index_allocator_mut().reserve();
        assert!(!storage.insert(b, 2).unwrap());

        let c = storage.index_allocator_mut().reserve();
        assert!(!storage.insert(c, 3).unwrap());

        assert_eq!(storage.get(a), Some(&1));
        assert_eq!(storage.get(b), Some(&2));
        assert_eq!(storage.get(c), Some(&3));

        assert_eq!(storage.buffer_len(), 3);

        storage.remove_recycle(a);
        storage.remove_recycle(b);

        let d = storage.index_allocator_mut().reserve();
        assert!(!storage.insert(d, 4).unwrap());

        let e = storage.index_allocator_mut().reserve();
        assert!(!storage.insert(e, 4).unwrap());
        assert!(storage.insert(e, 7).unwrap());

        assert_eq!(storage.get(d), Some(&4));
        assert_eq!(storage.get(e), Some(&7));

        assert!(storage.get(a).is_none());
        assert!(storage.insert(a, 8).is_err());

        assert_eq!(storage.buffer_len(), 3);
    }
}
