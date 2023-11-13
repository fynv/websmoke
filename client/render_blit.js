const shader_code =`

@group(0) @binding(0)
var uTexSmoke: texture_2d<f32>;

@group(0) @binding(1)
var uSampler: sampler;

struct VSOut 
{
    @builtin(position) Position: vec4f,    
};


@vertex
fn vs_main(@builtin(vertex_index) vertId: u32) -> VSOut
{
    var vsOut: VSOut;
    let grid = vec2(f32((vertId<<1)&2), f32(vertId & 2));
    let pos_proj = grid * vec2(2.0, 2.0) + vec2(-1.0, -1.0);        
    vsOut.Position = vec4(pos_proj, 0.0, 1.0);
    return vsOut;
}

@fragment
fn fs_main(@builtin(position) coord_pix: vec4f) -> @location(0) vec4f
{
    let icoord2d = vec2i(coord_pix.xy);
    let size = textureDimensions(uTexSmoke);
    let UV = (vec2f(icoord2d)+0.5)/vec2f(size);
    return textureSampleLevel(uTexSmoke, uSampler, UV, 0);
}
`;

function GetPipeline(view_format)
{
    if (!("render_blit" in engine_ctx.cache.pipelines))
    {
        const pipelineLayoutDesc = { bindGroupLayouts: [ engine_ctx.cache.bindGroupLayouts.frame  ] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        
        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: []
        };

        const colorState = {
            format:  view_format,           
            blend: {
                color: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
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
            frontFace: 'cw',
            cullMode: 'none',
            topology: 'triangle-list'
        };

        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive
        };

        engine_ctx.cache.pipelines.render_blit = engine_ctx.device.createRenderPipeline(pipelineDesc);

    }

    return engine_ctx.cache.pipelines.render_blit;

}


export function RenderBlit(passEncoder, bind_group_frame, target)
{
    let pipeline = GetPipeline(target.view_format);

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group_frame);    
    passEncoder.draw(3, 1);
}


