#version 450

layout (location = 0) in vec2 tex_coord;

layout (location = 0) out vec4 out_color;

layout (set = 0, binding = 0) uniform sampler2D depth_sampler;

float linearize_depth(float depth)
{
    float n = 1.0;
    float f = 96.0;
    float z = depth;
    return (2.0 * n) / (f + n - z * (f - n));
}

void main()
{
    float depth = texture(depth_sampler, tex_coord).r;
    out_color = vec4(vec3(1.0-linearize_depth(depth)), 1.0);
}