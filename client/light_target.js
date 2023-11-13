export class LightTarget
{
    constructor(width, height)
    {
        this.width = width;
        this.height = height;

        this.tex_alpha = engine_ctx.device.createTexture({
            size: { width, height},
            dimension: "2d",
            format: 'r8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        this.view_alpha =  this.tex_alpha.createView();

        this.sampler = engine_ctx.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: "linear"           
        });

        if (!("light_alpha" in engine_ctx.cache.bindGroupLayouts))
        {
            engine_ctx.cache.bindGroupLayouts.light_alpha = engine_ctx.device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture:{
                            viewDimension: "2d",                           
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler:{}
                    }
                ]
            });
        }

        const bindGroupLayout = engine_ctx.cache.bindGroupLayouts.light_alpha;
        this.bind_group = engine_ctx.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.view_alpha 
                },
                {
                    binding: 1,
                    resource: this.sampler
                },           
            ]
        });

    }

}
