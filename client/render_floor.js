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
var uTex: texture_2d<f32>;

@group(1) @binding(1)
var uSampler1: sampler;

@group(2) @binding(0)
var<uniform> uLightCamera: Camera;

@group(3) @binding(0)
var uTexLight: texture_2d<f32>;

@group(3) @binding(1)
var uSampler2: sampler;

struct VSIn 
{
    @builtin(vertex_index) vertId: u32,      
};

struct VSOut 
{
    @builtin(position) Position: vec4f,
    @location(0) uv: vec2f,        
    @location(1) world_pos: vec3f,
};


const c_uv = array(
    vec2(0.0, 0.0),
    vec2(20.0, 0.0),
    vec2(0.0, 20.0),
    vec2(20.0, 20.0),
    vec2(0.0, 20.0),
    vec2(20.0, 0.0)
);

const c_pos = array(
    vec3(-20.0, 0.0, -20.0),
    vec3(20.0, 0.0, -20.0),
    vec3(-20.0, 0.0, 20.0),
    vec3(20.0, 0.0, 20.0),
    vec3(-20.0, 0.0, 20.0),
    vec3(20.0, 0.0, -20.0)
);

@vertex
fn vs_main(input: VSIn) -> VSOut
{    
    let uv = c_uv[input.vertId];
    let pos = c_pos[input.vertId];

    let pos_view =  uCamera.viewMat * vec4(pos, 1.0);
    var pos_proj = uCamera.projMat * pos_view;
    pos_proj.z = (pos_proj.z + pos_proj.w) * 0.5;    

    var out: VSOut;
    out.Position = pos_proj;
    out.uv = uv;
    out.world_pos = pos;
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2f, @location(1) world_pos: vec3f) -> @location(0) vec4f
{
    let pos_light = uLightCamera.viewMat * vec4(world_pos, 1.0);
    var pos_light_proj =  uLightCamera.projMat * pos_light;
    pos_light_proj*=1.0/pos_light_proj.w;

    let dis_light = length(pos_light);
    let dis_rate = 50.0/(dis_light*dis_light);

    let lightUV = vec2((pos_light_proj.x + 1.0)*0.5, (1.0 - pos_light_proj.y)*0.5);

    var lightAlpha = 0.0;
    if (lightUV.x>0.0 && lightUV.x<1.0 && lightUV.y>0.0 && lightUV.y<1.0)
    {
        lightAlpha = textureSampleLevel(uTexLight, uSampler2, lightUV, 0).x;
    }

    let col = textureSampleLevel(uTex, uSampler1, uv, 0).xyz;
        
    let light_dir = normalize(-pos_light.xyz);
    let light_dir_world = uLightCamera.invViewMat * vec4(light_dir, 0.0);
    let dot_light_norm = max(light_dir_world.y, 0.0);
    let col_shaded = vec3(1.0, 1.0, 0.5) * col * dis_rate * (1.0-lightAlpha) * dot_light_norm;

    return vec4(col_shaded, 1.0);
}
`;

function GetPipeline(view_format, msaa)
{
    if (!("render_floor" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];
    
        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout, engine_ctx.cache.bindGroupLayouts.render_floor, camera_layout, engine_ctx.cache.bindGroupLayouts.light_alpha] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        const depthStencil = {
            depthWriteEnabled: true,
            depthCompare: 'less-equal',
            format: 'depth32float'
        };

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: []
        };

        const colorState = {
            format: view_format,                    
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
    
            primitive,
            depthStencil
        };

        if (msaa)
        {
            pipelineDesc.multisample ={
                count: 4,
            };
        }

        engine_ctx.cache.pipelines.render_floor = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }
    return engine_ctx.cache.pipelines.render_floor;
}

export function RenderFloor(passEncoder, camera, light, bind_group_floor, target)
{
    let pipeline = GetPipeline(target.view_format, target.msaa);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, bind_group_floor);
    passEncoder.setBindGroup(2, light.lightCamera.bind_group);
    passEncoder.setBindGroup(3, light.lightTarget.bind_group);    
    passEncoder.draw(6);    

}


