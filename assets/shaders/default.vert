#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_tex_coord;
layout(location = 2) out vec3 out_view_direction;
layout(location = 3) out vec3 out_light_direction;
layout(location = 4) out vec3 out_light_color;
layout(location = 5) out vec4 out_shadow_coord;

// per frame
layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 projection;
    // TODO if anything bad happens listen to this guy https://stackoverflow.com/questions/38172696/should-i-ever-use-a-vec3-inside-of-a-uniform-buffer-or-shader-storage-buffer-o
    vec3 camera_position;
} camera_uniform;

struct Light {
    vec4 position;
    vec4 color;
    // view projection matrix from the light's perspective
    mat4 view_projection;
    float intensity;
};

layout(set = 0, binding = 1) uniform LightsUniform {
    Light lights[10];
} lights_uniform;

// per primitive
layout(set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    mat4 normal;
} model_uniform;

// IDK
const mat4 bias_matrix = mat4(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);

void main() {
    Light light = lights_uniform.lights[0];

    // Fixes non-uniform scalings
    out_normal = vec3(model_uniform.normal * vec4(normalize(normal), 1.0));
    out_tex_coord = tex_coord;

    gl_Position = camera_uniform.projection * camera_uniform.view * model_uniform.model * vec4(position, 1.0);

    out_light_direction = normalize(light.position.xyz - position.xyz);
    out_light_color = light.color.rgb;
    out_view_direction = normalize(camera_uniform.camera_position - position.xyz);

//    out_shadow_coord = (bias_matrix * light.view_projection * model_uniform.model) * vec4(position, 1.0);
    out_shadow_coord = light.view_projection * vec4(position, 1.0);
}