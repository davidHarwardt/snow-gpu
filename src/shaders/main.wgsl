
struct InstanceInput {
    @location(2) position: vec4<f32>,
    @location(3) velocity: vec4<f32>,
    @location(4) scale: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
}

struct ShaderData {
    time: f32,
    dt: f32,
    aspect: f32,
}

@group(0) @binding(0)
var<uniform> data: ShaderData;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_pos = vec4<f32>(((model.position * instance.scale * 0.002) + instance.position.xyz) * vec3<f32>(data.aspect, 1.0, 1.0), 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

