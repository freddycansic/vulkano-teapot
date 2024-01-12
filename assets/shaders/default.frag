#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 tex_coord;
layout(location = 2) in vec3 view_direction;
layout(location = 3) in vec3 light_direction;
layout(location = 4) in vec3 light_color;
layout(location = 5) in vec4 shadow_coord;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 2) uniform sampler2D shadowmap_sampler;

layout(set = 1, binding = 1) uniform sampler2D texture_sampler;

const float ambient_strength = 0.1;

float texture_project(vec4 shadow_coord, vec2 offset) {
    float shadow = 1.0;
    if ( shadow_coord.z > -1.0 && shadow_coord.z < 1.0 )
    {
        float distance = texture(shadowmap_sampler, shadow_coord.st + offset).r;
        if ( shadow_coord.w > 0.0 && distance < shadow_coord.z )
        {
            shadow = ambient_strength;
        }
    }

    return shadow;
}

float filter_PCF(vec4 sc) {
    ivec2 texture_dimensions = textureSize(shadowmap_sampler, 0);
    float scale = 1.5;
    float dx = scale * 1.0 / float(texture_dimensions.x);
    float dy = scale * 1.0 / float(texture_dimensions.y);

    float shadowFactor = 0.0;
    int count = 0;
    int range = 1;

    for (int x = -range; x <= range; x++)
    {
        for (int y = -range; y <= range; y++)
        {
            shadowFactor += texture_project(sc, vec2(dx*x, dy*y));
            count++;
        }

    }
    return shadowFactor / count;
}

void main() {
    float shadow = filter_PCF(shadow_coord / shadow_coord.w);
//    float shadow = texture_project(shadow_coord / shadow_coord.w, vec2(0.0));

    vec3 color = texture(texture_sampler, tex_coord).rgb;

    // ambient
    vec3 ambient = ambient_strength * light_color;

    // diffuse
    // how close is the angle of incidence to the normal?
    float incidence_angle = max(dot(normal, light_direction), 0.0);
    vec3 diffuse = incidence_angle * light_color;

    // specular
    // how close is the direction of reflected light to the direction from the fragment to the eye?
    float specular_strength = 0.5;
    vec3 reflection_direction = reflect(-light_direction, normal);

    float shininess = 64;
    float specularity = pow(max(dot(view_direction, reflection_direction), 0.0), shininess);
    vec3 specular = specular_strength * specularity * light_color;

//    vec4(ambient + diffuse + specular, 1.0);

    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;

    out_color = vec4(lighting, 1.0);
//    out_color = calculate_lighting(color) * color * shadow;
}