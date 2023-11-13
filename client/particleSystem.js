import {CalcKey} from "./calc_key.js"
import {KeyReduction} from "./reduce_key.js"
import {ResetIndices} from "./reset_indices.js"
import {RadixSort} from "./radix_sort.js"
import {Integrate} from "./integrate.js"
import { Vector3 } from "./engine/math/Vector3.js"

const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

export class ParticleSystem
{
    constructor(numParticles)
    {
        this.numParticles = numParticles;
        this._initialize();
    }

    _initialize()
    {
        this.cursor_angle = 0.0;
        this.cursor_pos = new Vector3(0.0, 1.0, 0.0);
        this.cursor_pos_lag = new Vector3(0.0, 1.0, 0.0);
        this.emitterIndex = 0;

        this.hPos = new Float32Array(this.numParticles * 4);
        this.hVel = new Float32Array(this.numParticles * 4);
        this.tex_noise =  engine_ctx.device.createTexture({
            size: { width: 64, height: 64, depthOrArrayLayers: 64 },
            dimension: "3d",
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.sampler = engine_ctx.device.createSampler({ addressModeU:"repeat", addressModeV: "repeat", addressModeW: "repeat",  magFilter: "linear", minFilter:"linear"});

        {
            let noise_data = new Uint8Array(64*64*64*4);
            for (let i=0; i<64*64*64; i++)
            {
                noise_data[i*4] = Math.random() * 255.0 + 0.5;
                noise_data[i*4 + 1] = Math.random() * 255.0 + 0.5;
                noise_data[i*4 + 2] = Math.random() * 255.0 + 0.5;
                noise_data[i*4 + 3] = Math.random() * 255.0 + 0.5;
            }
            engine_ctx.queue.writeTexture(
                { texture:this.tex_noise },
                noise_data.buffer,
                { bytesPerRow: 64 * 4, rowsPerImage:  64 },
                { width: 64, height: 64, depthOrArrayLayers: 64 },
            );
        }
        
        let memSize = this.numParticles * 4 * 4;
        this.dPos = engine_ctx.createBuffer0(memSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.dVel = engine_ctx.createBuffer0(memSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

        this.dConstantHalfVec = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.dKey = [];
        
        let buf_size = this.numParticles;    
        {
            let buf = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE);
            this.dKey.push(buf);
        }
        while(buf_size>1)
        {
            buf_size = Math.floor((buf_size + workgroup_size_2x - 1)/workgroup_size_2x);
            let buf = engine_ctx.createBuffer0(buf_size*4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            this.dKey.push(buf);
        }

        this.dKeyMinMax = engine_ctx.createBuffer0(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
        this.dIndices =  engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE);
        this.dIndices1 = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);        
        this.dConstantSort = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        this.dBufScan1 =[];
        this.dBufScan2 =[];
        this.dBufScanSizes = [];
        buf_size = this.numParticles;
        while (buf_size>0)
        {
            let buf1 = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE);
            let buf2 = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE);
            this.dBufScan1.push(buf1);
            this.dBufScan2.push(buf2);
            this.dBufScanSizes.push(buf_size);
            buf_size = Math.floor((buf_size + workgroup_size_2x - 1)/workgroup_size_2x) - 1;
        }

        //////////////////////////////////////////////////////////

        engine_ctx.cache.bindGroupLayouts.integrate = engine_ctx.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }                    
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }                    
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler:{}
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture:{
                        viewDimension: "3d"
                    }
                }
            ]
        });

        engine_ctx.cache.bindGroupLayouts.render_smoke = engine_ctx.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer:{
                        type: "read-only-storage"
                    }                    
                }
            ]
        });
        
        engine_ctx.cache.bindGroupLayouts.calc_key = engine_ctx.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "uniform"
                    }
                   
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }                    
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                }
            ]
        });

        engine_ctx.cache.bindGroupLayouts.key_reduction = engine_ctx.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                }
            ]
        });

        engine_ctx.cache.bindGroupLayouts.reset_indices = engine_ctx.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                }
            ]
        });

        let layout_entries_radix_scan1 = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
        ];

        if (this.dBufScan1.length>1)
        {
            layout_entries_radix_scan1.push({
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            });

            layout_entries_radix_scan1.push({
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            });
        }

        let bindGroupLayoutRadixScan1 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan1 });
        if (this.dBufScan1.length>1)
        {
            engine_ctx.cache.bindGroupLayouts.radixScan1B = bindGroupLayoutRadixScan1;    
        }
        else
        {
            engine_ctx.cache.bindGroupLayouts.radixScan1A = bindGroupLayoutRadixScan1;
        }

        let layout_entries_radix_scan2 = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
        ];

        engine_ctx.cache.bindGroupLayouts.radixScan2A =engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan2 });    

        layout_entries_radix_scan2.push({
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        });
    
        layout_entries_radix_scan2.push({
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        });

        engine_ctx.cache.bindGroupLayouts.radixScan2B = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan2 });

        engine_ctx.cache.bindGroupLayouts.radixScan3 = engine_ctx.device.createBindGroupLayout({ 
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                }
            ]                    
        });
    
        engine_ctx.cache.bindGroupLayouts.radixScatter =  engine_ctx.device.createBindGroupLayout({ 
            entries: [  
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                }
            ]
        });

        ///////////////////////////////////////////////////////////////////////////////////
        let group_entries_integrate = [
            {
                binding: 0,
                resource:{
                    buffer: this.dPos            
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dVel
                }
            },
            {
                binding: 2,
                resource: this.sampler
            },
            {
                binding: 3,
                resource: this.tex_noise.createView()
            }
        ];

        this.bind_group_integrate = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.integrate,  entries: group_entries_integrate});

        let group_entries_render_smoke = [
            {
                binding: 0,
                resource:{
                    buffer: this.dPos            
                }
            }
        ];
        this.bind_group_render_smoke = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.render_smoke, entries: group_entries_render_smoke});

        let group_entries_calc_key = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstantHalfVec            
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dPos
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dKey[0]
                }
            },
        ];

        this.bind_group_calc_key = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.calc_key, entries: group_entries_calc_key});

        this.bind_group_key_reduction = [];
        for (let i=0; i<this.dKey.length-1; i++)
        {
            let group_entries = [
                {
                    binding: 0,
                    resource:{
                        buffer: this.dKey[i]            
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dKey[i+1]
                    }
                }
            ];
            this.bind_group_key_reduction.push(engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.key_reduction, entries: group_entries}));
        }

        let group_entries_reset_indices = [
            {
                binding: 0,
                resource:{
                    buffer: this.dIndices            
                }
            },           
        ];

        this.bind_group_reset_indices = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.reset_indices, entries: group_entries_reset_indices});


        this.bind_group_radix_scan1 = new Array(2);
        this.bind_group_radix_scatter = new Array(2);

        let group_entries_radix_scan10 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstantSort
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dKeyMinMax
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dKey[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dIndices
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
        ];

        if (this.dBufScan1.length>1)
        {
            group_entries_radix_scan10.push({
                binding: 6,
                resource:{
                    buffer: this.dBufScan1[1]
                }
            });

            group_entries_radix_scan10.push({
                binding: 7,
                resource:{
                    buffer: this.dBufScan2[1]
                }
            });
        }

        this.bind_group_radix_scan1[0] = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan1, entries: group_entries_radix_scan10}); 

        let group_entries_radix_scatter0 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dIndices    
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dIndices1
                }
            },
        ];

        this.bind_group_radix_scatter[0] = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.radixScatter, entries: group_entries_radix_scatter0});

        let group_entries_radix_scan11 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstantSort
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dKeyMinMax
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dKey[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer:  this.dIndices1
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
        ];

        if (this.dBufScan1.length>1)
        {
            group_entries_radix_scan11.push({
                binding: 6,
                resource:{
                    buffer: this.dBufScan1[1]
                }
            });

            group_entries_radix_scan11.push({
                binding: 7,
                resource:{
                    buffer: this.dBufScan2[1]
                }
            });
        }

        this.bind_group_radix_scan1[1] = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan1, entries: group_entries_radix_scan11}); 

        let group_entries_radix_scatter1 = [
            {
                binding: 0,
                resource:{
                    buffer: this.dIndices1    
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dBufScan1[0]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dBufScan2[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dIndices
                }
            },
        ];

        this.bind_group_radix_scatter[1] = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.radixScatter, entries: group_entries_radix_scatter1});

        this.bind_group_radix_scan2 = [];
        for (let i=1; i<this.dBufScan1.length; i++)
        {
            let group_entries_radix_scan = [            
                {
                    binding: 0,
                    resource:{
                        buffer: this.dBufScan1[i]
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dBufScan2[i]
                    }
                },
            ];

            if (i<this.dBufScan1.length-1)
            {
                group_entries_radix_scan.push({
                    binding: 2,
                    resource:{
                        buffer: this.dBufScan1[i+1]
                    }
                });
    
                group_entries_radix_scan.push({
                    binding: 3,
                    resource:{
                        buffer: this.dBufScan2[i+1]
                    }
                });

                this.bind_group_radix_scan2.push(engine_ctx.device.createBindGroup({ layout:  engine_ctx.cache.bindGroupLayouts.radixScan2B, entries: group_entries_radix_scan}));
            }
            else
            {
                this.bind_group_radix_scan2.push(engine_ctx.device.createBindGroup({ layout:  engine_ctx.cache.bindGroupLayouts.radixScan2A, entries: group_entries_radix_scan}));
            }
        }

        this.bind_group_radix_scan3 = [];
        for (let i=0; i < this.dBufScan1.length - 1; i++)
        {
            let group_entries_radix_scan = [            
                {
                    binding: 0,
                    resource:{
                        buffer: this.dBufScan1[i]
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dBufScan2[i]
                    }
                },
                {
                    binding: 2,
                    resource:{
                        buffer: this.dBufScan1[i + 1]
                    }
                },
                {
                    binding: 3,
                    resource:{
                        buffer: this.dBufScan2[i + 1]
                    }
                }
            ];
            this.bind_group_radix_scan3.push(engine_ctx.device.createBindGroup({ layout:  engine_ctx.cache.bindGroupLayouts.radixScan3, entries: group_entries_radix_scan}));
        }

    }

    _initCubeRandom()
    {
        for (let i=0; i < this.numParticles; i++)
        {
            this.hPos[i*4] = Math.random() * 2.0 - 1.0;
            this.hPos[i*4 + 1] = Math.random() * 2.0;
            this.hPos[i*4 + 2] = Math.random() * 2.0 - 1.0;
            this.hPos[i*4 + 3] = 0.0;

            this.hVel[i*4] = 0.0;
            this.hVel[i*4 + 1] = 0.0;
            this.hVel[i*4 + 2] = 0.0;
            this.hVel[i*4 + 3] = 100.0;
        }
    }

    reset()
    {
        this._initCubeRandom();
        engine_ctx.queue.writeBuffer(this.dPos, 0, this.hPos.buffer, 0, this.hPos.length * 4);
        engine_ctx.queue.writeBuffer(this.dVel, 0, this.hVel.buffer, 0, this.hVel.length * 4);

        let commandEncoder = engine_ctx.device.createCommandEncoder();  
        ResetIndices(commandEncoder, this);
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);

    }

    update()
    {
        this.cursor_angle += 0.01;        
        this.cursor_pos.x = Math.sin(this.cursor_angle) * 1.5;
        this.cursor_pos.y = 1.5 + Math.sin(this.cursor_angle*1.3);
        this.cursor_pos.z = Math.cos(this.cursor_angle) * 1.5;
        this.cursor_pos_lag.lerp(this.cursor_pos, 0.1);

        let count = 512;
        let r = 0.25;
        for (let i=0; i<count; i++)
        {
            let x = new Vector3();
            x.randomDirection();
            x.multiplyScalar (r);            
            let p = this.cursor_pos_lag.clone();
            p.add(x);            

            let lt = 64.0 + Math.random() * 6.4;

            let index = this.emitterIndex + i;
            this.hPos[index * 4] = p.x;
            this.hPos[index * 4 + 1] = p.y;
            this.hPos[index * 4 + 2] = p.z;
            this.hPos[index * 4 + 3] = 0.0;
            this.hVel[index * 4] = 0.0;
            this.hVel[index * 4 + 1] = 0.0;
            this.hVel[index * 4 + 2] = 0.0;
            this.hVel[index * 4 + 3] = 0.0;
        }        
        engine_ctx.queue.writeBuffer(this.dPos, this.emitterIndex * 4 * 4, this.hPos.buffer, this.emitterIndex * 4 * 4, count*4*4);
        engine_ctx.queue.writeBuffer(this.dVel, this.emitterIndex * 4 * 4, this.hVel.buffer, this.emitterIndex * 4 * 4, count*4*4);
        this.emitterIndex = (this.emitterIndex + count)%this.numParticles;

        let commandEncoder = engine_ctx.device.createCommandEncoder();    
        Integrate(commandEncoder, this);
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);
    }

    sort(halfVector)
    {
        {
            const uniform = new Float32Array(4);
            uniform[0] = halfVector.x;
            uniform[1] = halfVector.y;
            uniform[2] = halfVector.z;
            engine_ctx.queue.writeBuffer(this.dConstantHalfVec, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }
        
        let commandEncoder = engine_ctx.device.createCommandEncoder();    
        CalcKey(commandEncoder, this);
        KeyReduction(commandEncoder, this);
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);
        
        RadixSort(this);
               

    }    

}