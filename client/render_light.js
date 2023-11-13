const shader_code =`
struct Camera
{
    projMat: mat4x4f, 
    viewMat: mat4x4f,
    invProjMat: mat4x4f,
    invViewMat: mat4x4f,
    eyePos: vec4f,
    scissor: vec4f
};

@group(0) @binding(0)
var<uniform> uCamera: Camera;

@group(1) @binding(0)
var<storage, read> bPos : array<vec4f>;

const radius = 0.05;
const alpha = 0.1;

struct VSIn 
{
    @builtin(vertex_index) vertId: u32,    
    @location(0) index : u32,    
};

struct VSOut 
{
    @builtin(position) Position: vec4f,
    @location(0) uv: vec2f,        
};

const c_uv = array(
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0)
);

@vertex
fn vs_main(input: VSIn) -> VSOut
{    
    let uv = c_uv[input.vertId];  
    let pos = vec4(bPos[input.index].xyz, 1.0);
    let center_view = uCamera.viewMat * pos;

    let d = radius*2.0;
    let pos_view = center_view + vec4(d*(uv-0.5), 0.0, 0.0);
    var pos_proj = uCamera.projMat * pos_view;
    pos_proj.z = (pos_proj.z + pos_proj.w) * 0.5;

    var out: VSOut;
    out.Position = pos_proj;
    out.uv = uv;    

    return out;
}


@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) f32
{    
    var N: vec3f;
    N = vec3(uv * 2.0 -1.0, 0.0);
    let mag = length(N.xy);
    if (mag>1.0)
    {
        discard;
    }
    N.z = sqrt(1-mag*mag);
    
    return alpha * (1.0 - mag * mag);
}
`;

function GetPipeline()
{
    if (!("render_light" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];

        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout, engine_ctx.cache.bindGroupLayouts.render_smoke] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        let vertex_bufs = [
            {            
                arrayStride: 4,
                stepMode: 'instance',
                attributes: [
                  {                
                    shaderLocation: 0,
                    offset: 0,
                    format: 'uint32',
                  },             
                ],
            }
        ];

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: vertex_bufs
        };

        const colorState = {
            format: "r8unorm",        
            blend: {
                color: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src"
                },
                alpha: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                }
            },
            writeMask: GPUColorWrite.ALL
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [colorState]
        };
    
        const primitive = {
            frontFace: 'ccw',
            cullMode:  "none",
            topology: 'triangle-list'
        };

        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive
        };

        engine_ctx.cache.pipelines.render_light = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }

    return engine_ctx.cache.pipelines.render_light;
}


export function RenderLight(passEncoder, camera, psystem, start, count)
{
    let pipeline = GetPipeline();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, psystem.bind_group_render_smoke);
    passEncoder.setVertexBuffer(0, psystem.dIndices);    
    passEncoder.draw(6, count, 0, start);    
}


