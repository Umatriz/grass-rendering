use std::{any::type_name, marker::PhantomData};

use bevy_app::{App, Plugin};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    world::{Mut, World},
};
use crossbeam_channel::{Receiver, Sender};
use tracing::error;

use crate::{
    Result,
    dense_storage::{DenseStorage, Index},
};

use super::{CleanUp, render_context::destroy_render_context};

pub struct RenderAssetsPlugin;

impl Plugin for RenderAssetsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(DeletionQueue::new()).add_systems(
            CleanUp,
            delete_queued_resources.before(destroy_render_context),
        );
    }
}

pub trait RegisterRenderAssetAppExt {
    fn app_mut(&mut self) -> &mut App;
    fn register_render_asset<T: RenderAsset>(&mut self) -> &mut App {
        let app = self.app_mut();
        let queue = app.world_mut().get_resource::<DeletionQueue>().expect("`DeletionQueue` is not found while registering a render asset. Make sure you're calling `register_render_asset` after you've added `RenderAssetsPlugin`!").sender();
        app.insert_resource(RenderAssets::<T>::new(queue));
        app
    }
}

type DeletionFn = Box<dyn FnMut(&mut World) + Send + Sync + 'static>;

#[derive(Resource)]
pub struct DeletionQueue {
    queue: Receiver<DeletionFn>,
    sender: Sender<DeletionFn>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        let (s, r) = crossbeam_channel::unbounded();
        Self {
            queue: r,
            sender: s,
        }
    }

    pub fn sender(&self) -> DeletionQueueSender {
        DeletionQueueSender {
            sender: self.sender.clone(),
        }
    }
}

fn delete_queued_resources(world: &mut World) {
    let queue = world
        .remove_resource::<DeletionQueue>()
        .expect("`DeletionQueue` does not exist in the world");

    while let Ok(mut fun) = queue.queue.recv() {
        (fun)(world);
    }
}

pub struct DeletionQueueSender {
    sender: Sender<DeletionFn>,
}

impl DeletionQueueSender {
    pub fn queue_deletion(&self, deletion_fn: DeletionFn) -> Result<()> {
        self.sender.send(deletion_fn)?;
        Ok(())
    }
}
pub trait RenderAsset: Send + Sync + 'static {
    fn destroy(self, world: &mut World) -> Result<()>;
}

#[derive(Resource)]
pub struct RenderAssets<T> {
    storage: DenseStorage<T>,
    deletion_queue: DeletionQueueSender,
}

impl<T: RenderAsset> RenderAssets<T> {
    fn new(queue: DeletionQueueSender) -> Self {
        Self {
            storage: DenseStorage::default(),
            deletion_queue: queue,
        }
    }

    pub fn add(&mut self, asset: T) -> Handle<T> {
        let index = self.storage.push(asset);
        let handle = Handle::new(index);

        let del_handle = handle.clone();
        let fun = move |world: &mut World| {
            world.resource_scope(|world: &mut World, mut res: Mut<'_, RenderAssets<T>>| {
                if let Some(remove) = res.remove(del_handle.clone()) {
                    remove.destroy(world).inspect_err(|e| {
                        error!(
                            "failed to destroy resource of type {}. Error: {}",
                            type_name::<T>(),
                            e
                        )
                    });
                } else {
                    error!("failed to delete resource of type {}", type_name::<T>())
                };
            });
        };
        self.deletion_queue.queue_deletion(Box::new(fun));

        handle
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.storage.get(handle.index)
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.storage.get_mut(handle.index)
    }

    /// Removes the resource from the storage but **does not destroy it**.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        self.storage.remove_recycle(handle.index)
    }
}

pub struct Handle<T> {
    index: Index,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index.clone(),
            _marker: self._marker.clone(),
        }
    }
}

impl<T> Handle<T> {
    fn new(index: Index) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }
}
