const workgroup_size = 64;

const shader_code = `
const deltaTime = 0.5;
const noiseFreq = 0.1;
const noiseAmp = 0.001;
const globalDamping = 0.99;

@group(0) @binding(0)
var<storage, read_write> bPos : array<vec4f>;

@group(0) @binding(1)
var<storage, read_write> bVel : array<vec4f>;

@group(0) @binding(2) 
var uSampler: sampler;

@group(0) @binding(3)
var uTex: texture_3d<f32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bPos))
    {
        return;
    }

    let posData = bPos[idx];
    let velData = bVel[idx];

    var pos = posData.xyz;
    var vel = velData.xyz;

    var age = posData.w;
    let lifetime = velData.w;


    if (age < lifetime)
    {
        age += deltaTime;
    }
    else
    {
        age = lifetime;
    }
    
    let noiseP = pos * noiseFreq;
    let noise =  textureSampleLevel(uTex, uSampler, noiseP, 0.0).xyz * 2.0 - 1.0;
    vel += noise * noiseAmp;
    pos += vel * deltaTime;
    vel*= globalDamping;

    bPos[idx] = vec4(pos, age);
    bVel[idx] = vec4(vel, lifetime);
}
`;

function GetPipeline()
{
    if (!("integrate" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.integrate];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.integrate = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.integrate;
}


export function Integrate(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups =  Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_integrate;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

