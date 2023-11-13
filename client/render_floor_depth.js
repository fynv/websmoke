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

struct VSIn 
{
    @builtin(vertex_index) vertId: u32,      
};

struct VSOut 
{
    @builtin(position) Position: vec4f,   
};

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
    let pos = c_pos[input.vertId];

    let pos_view =  uCamera.viewMat * vec4(pos, 1.0);
    var pos_proj = uCamera.projMat * pos_view;
    pos_proj.z = (pos_proj.z + pos_proj.w) * 0.5;    

    var out: VSOut;
    out.Position = pos_proj;    
    return out;
}

@fragment
fn fs_main()
{
}
`;

function GetPipeline(msaa)
{
    if (!("render_floor_depth" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];

        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout] };
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

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: []
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

        engine_ctx.cache.pipelines.render_floor_depth = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }
    return engine_ctx.cache.pipelines.render_floor_depth;
}

export function RenderFloorDepth(passEncoder, camera, target)
{
    let pipeline = GetPipeline(target.msaa);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);        
    passEncoder.draw(6);    
}

