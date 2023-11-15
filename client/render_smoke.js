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

@group(2) @binding(0)
var<uniform> uLightCamera: Camera;

@group(3) @binding(0)
var uTexLight: texture_2d<f32>;

@group(3) @binding(1)
var uSampler: sampler;

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
    @location(1) world_pos: vec3f,
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
    out.world_pos = pos.xyz;

    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2f, @location(1) world_pos: vec3f) -> @location(0) vec4f
{    
    var N: vec3f;
    N = vec3(uv * 2.0 -1.0, 0.0);
    let mag = length(N.xy);
    if (mag>1.0)
    {
        discard;
    }
    N.z = sqrt(1-mag*mag);

    let pos_world = vec4(world_pos, 1.0) + uCamera.invViewMat*vec4(N * radius, 0.0);
    let pos_light = uLightCamera.viewMat * pos_world;
    var pos_light_proj =  uLightCamera.projMat * pos_light;    
    pos_light_proj*=1.0/pos_light_proj.w;

    let dis_light = length(pos_light);
    let dis_rate = 100.0/(dis_light*dis_light);

    let lightUV = vec2((pos_light_proj.x + 1.0)*0.5, (1.0 - pos_light_proj.y)*0.5);
    var lightAlpha = 0.0;
    if (lightUV.x>0.0 && lightUV.x<1.0 && lightUV.y>0.0 && lightUV.y<1.0)
    {
        lightAlpha = textureSampleLevel(uTexLight, uSampler, lightUV, 0).x;
    }

    let col = vec3(1.0, 1.0, 0.5) * vec3(0.7, 0.7, 0.7) * dis_rate * (1.0-lightAlpha);
    let a = alpha * (1.0 - mag * mag);
    return vec4(col*a, a);
}
`;

function GetPipeline(view_format, msaa, invertedView)
{
    if (!("render_smoke" in engine_ctx.cache.pipelines))
    {
        engine_ctx.cache.pipelines.render_smoke = {};
    }

    if (!(invertedView in engine_ctx.cache.pipelines.render_smoke))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];    
    
        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout, engine_ctx.cache.bindGroupLayouts.render_smoke, camera_layout, engine_ctx.cache.bindGroupLayouts.light_alpha] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        const depthStencil = {
            depthWriteEnabled: false,
            depthCompare: 'less-equal',
            format: 'depth32float'
        };

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
            format: view_format,                    
            writeMask: GPUColorWrite.ALL
        };

        if (invertedView)
        {
            colorState.blend= {
                color: {
                    srcFactor: "one-minus-dst-alpha",
                    dstFactor: "one"
                },
                alpha: {
                    srcFactor: "one-minus-dst-alpha",
                    dstFactor: "one"
                }
            };
        }
        else
        {
            colorState.blend= {
                color: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                },
                alpha: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                }
            };
        }

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
    
            primitive,
            depthStencil
        };

        if (msaa)
        {
            pipelineDesc.multisample ={
                count: 4,
            };
        }

        engine_ctx.cache.pipelines.render_smoke[invertedView] = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }

    return engine_ctx.cache.pipelines.render_smoke[invertedView];
}

export function RenderSmoke(passEncoder, camera, light, psystem, target, start, count, invertedView)
{
    let pipeline = GetPipeline(target.view_format, target.msaa, invertedView);

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, psystem.bind_group_render_smoke);
    passEncoder.setBindGroup(2, light.lightCamera.bind_group);
    passEncoder.setBindGroup(3, light.lightTarget.bind_group);
    passEncoder.setVertexBuffer(0, psystem.dIndices);    
    passEncoder.draw(6, count, 0, start);    
}

