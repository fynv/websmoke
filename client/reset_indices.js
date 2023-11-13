const workgroup_size = 64;

const shader_code = `
@group(0) @binding(0)
var<storage, read_write> bIndices : array<u32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bIndices)) 
    {
        return;
    }

    bIndices[idx] = idx;
}
`;

function GetPipeline()
{
    if (!("reset_indices" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.reset_indices];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.reset_indices = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.reset_indices;
}


export function ResetIndices(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups = Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_reset_indices;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}


