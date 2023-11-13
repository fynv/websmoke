const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

const shader_code1 = `
@group(0) @binding(0)
var<storage, read> bKeyIn : array<f32>;

@group(0) @binding(1)
var<storage, read_write> bKeyOut : array<f32>;

var<workgroup> s_buf : array<f32, ${workgroup_size_2x}>;


@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bKeyIn);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx] = bKeyIn[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx + ${workgroup_size}] = bKeyIn[i];
    }

    workgroupBarrier();

    var stride = ${workgroup_size}u;
    while(stride>0)
    {
        if (threadIdx<stride)
        {
            i = threadIdx + stride  + blockIdx*${workgroup_size_2x};
            if (i<count)
            {
                s_buf[threadIdx] = min(s_buf[threadIdx], s_buf[threadIdx + stride]);
            }
        }
        stride = stride >> 1;
        workgroupBarrier();        
    }

    if (threadIdx==0)
    {
        bKeyOut[blockIdx] = s_buf[0];
    }
}
`;

const shader_code2 = `
@group(0) @binding(0)
var<storage, read> bKeyIn : array<f32>;

@group(0) @binding(1)
var<storage, read_write> bKeyOut : array<f32>;

var<workgroup> s_buf : array<f32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bKeyIn);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx] = bKeyIn[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx + ${workgroup_size}] = bKeyIn[i];
    }

    workgroupBarrier();

    var stride = ${workgroup_size}u;
    while(stride>0)
    {
        if (threadIdx<stride)
        {
            i = threadIdx + stride  + blockIdx*${workgroup_size_2x};
            if (i<count)
            {
                s_buf[threadIdx] = max(s_buf[threadIdx], s_buf[threadIdx + stride]);
            }
        }
        stride = stride >> 1;
        workgroupBarrier();        
    }

    if (threadIdx==0)
    {
        bKeyOut[blockIdx] = s_buf[0];
    }
}
`;

function GetPipeline1()
{
    if (!("key_reduction_1" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code1 });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.key_reduction];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.key_reduction_1 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.key_reduction_1;
}

function GetPipeline2()
{
    if (!("key_reduction_2" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code2 });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.key_reduction];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.key_reduction_2 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.key_reduction_2;
}


export function KeyReduction(commandEncoder, psystem)
{
    {
        let pipeline = GetPipeline1();
        const passEncoder = commandEncoder.beginComputePass();
        let count = psystem.numParticles;  
        for (let i=0; i<psystem.dKey.length-1; i++)
        {
            count = Math.floor((count + workgroup_size_2x - 1)/workgroup_size_2x);
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, psystem.bind_group_key_reduction[i]);
            passEncoder.dispatchWorkgroups(count, 1,1); 
        }
        passEncoder.end();
    }

    commandEncoder.copyBufferToBuffer(psystem.dKey[psystem.dKey.length-1], 0, psystem.dKeyMinMax, 0, 4);

    {
        let pipeline = GetPipeline2();
        const passEncoder = commandEncoder.beginComputePass();
        let count = psystem.numParticles;  
        for (let i=0; i<psystem.dKey.length-1; i++)
        {
            count = Math.floor((count + workgroup_size_2x - 1)/workgroup_size_2x);           
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, psystem.bind_group_key_reduction[i]);
            passEncoder.dispatchWorkgroups(count, 1,1); 
        }
        passEncoder.end();
    }

    commandEncoder.copyBufferToBuffer(psystem.dKey[psystem.dKey.length-1], 0, psystem.dKeyMinMax, 4, 4);

}



