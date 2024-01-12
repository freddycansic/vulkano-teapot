use cgmath::{Matrix4, SquareMatrix};

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/default.vert",
        // TODO linalg_type: "cgmath"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/default.frag",
    }
}

impl Default for vs::Light {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0].into(),
            color: [1.0, 1.0, 1.0, 1.0].into(),
            view_projection: Matrix4::identity().into(),
            intensity: 1.0,
        }
    }
}

pub mod offscreen_depth_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/offscreen_depth.vert"
    }
}

pub mod depth_texture_vertex {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/depth_texture.vert"
    }
}

pub mod depth_texture_fragment {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/depth_texture.frag"
    }
}
