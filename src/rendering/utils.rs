use ash::{Device, Instance, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};

pub fn create_shader_module(device: &Device, buf: &[u8]) -> vk::ShaderModule {
    let code: &[u32] = bytemuck::cast_slice(buf);
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);

    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}

pub fn create_buffer(
    device: &Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_location: MemoryLocation,
) -> (vk::Buffer, Allocation) {
    unsafe {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None).unwrap();
        let mem_requirements = device.get_buffer_memory_requirements(buffer);

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "Buffer allocation",
                requirements: mem_requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .unwrap();

        (buffer, allocation)
    }
}

pub fn create_image(
    device: &Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    memory_location: MemoryLocation,
) -> (vk::Image, Allocation) {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D::default().width(width).height(height).depth(1))
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(usage);

    let image = unsafe { device.create_image(&image_info, None).unwrap() };

    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let image_allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "image allocation",
            requirements: mem_req,
            location: memory_location,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();

    unsafe {
        device
            .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
            .unwrap()
    };

    (image, image_allocation)
}

pub fn transition_image_layout(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access_mask: vk::AccessFlags2,
    dst_access_mask: vk::AccessFlags2,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    image_aspect_mask: vk::ImageAspectFlags,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(image_aspect_mask)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let image_memory_barriers = &[barrier];
    let dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);
    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };
}

pub fn copy_buffer_to_image(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    width: u32,
    height: u32,
) {
    let regions = &[vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D::default().width(width).height(height).depth(1))];

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            src_buffer,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions,
        )
    };
}

pub fn single_time_commands(
    device: &Device,
    command_pool: &vk::CommandPool,
    queue: &vk::Queue,
    fun: impl FnOnce(&Device, vk::CommandBuffer),
) {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&allocate_info)
            .unwrap()
            .pop()
            .unwrap()
    };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap()
    };

    fun(device, command_buffer);

    unsafe { device.end_command_buffer(command_buffer).unwrap() };

    let buffers = &[command_buffer];
    let submit_info = &[vk::SubmitInfo::default().command_buffers(buffers)];
    unsafe {
        device
            .queue_submit(*queue, submit_info, vk::Fence::null())
            .unwrap()
    };
}

pub fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    unsafe {
        let mem_properties = instance.get_physical_device_memory_properties(physical_device);

        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i) > 0)
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return i;
            }
        }

        panic!("failed to find suitable memory type")
    }
}
