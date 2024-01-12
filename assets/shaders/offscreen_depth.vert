#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform ViewProjectionUniform
{
    mat4 view_projection;
} view_projection_uniform;

layout(set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
} model_uniform;

void main()
{
    gl_Position = view_projection_uniform.view_projection * model_uniform.model *  vec4(position, 1.0);
}