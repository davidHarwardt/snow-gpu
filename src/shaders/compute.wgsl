struct Instance {
    position: vec4<f32>,
    velocity: vec4<f32>,
    scale: f32,
}

struct ShaderData {
    time: f32,
    dt: f32,
    aspect: f32,
}

@group(0) @binding(0)
var<storage, read_write> instances: array<Instance>;

@group(1) @binding(0)
var<uniform> data: ShaderData;

fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

fn simple_noise(v: f32) -> f32 {
    return sin(v * 0.2) * 0.4
         + sin(v * 0.9) * 0.2
         + sin(v * 5.0) * 0.05
         + sin(v * 0.5) * 0.35;
}

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let padding = 0.1;

    var pos = instances[global_id.x].position.xyz;
    var vel = instances[global_id.x].velocity.xyz;
    var scale = instances[global_id.x].scale;
    
    pos = pos + (vel * data.dt);
    let wind_dir = simple_noise(data.time * 0.1) * 0.05;
    vel = vel + (vec3<f32>(wind_dir, -0.1, 0.0) * data.dt * (1.0 / scale));

    if(pos.y + padding < -1.0) {
        let r = rand(pos.xy);

        pos.x = (r * (2.0 + padding * 2.5) - 1.0 * (1.0 + padding * 2.5)) / data.aspect;
        pos.y = 1.0 + padding;
        vel = vec3<f32>(0.0);
    }

    // if((abs(pos.x) - padding) > 1.0 / data.aspect) {
    //     pos.x = -pos.x - sign(pos.x) * 0.01;
    // }

    instances[global_id.x].position = vec4<f32>(pos, 1.0);
    instances[global_id.x].velocity = vec4<f32>(vel, 1.0);
}

