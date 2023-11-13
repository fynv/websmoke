import { EngineContext } from "./engine/EngineContext.js"
import { CanvasContext } from "./engine/CanvasContext.js"
import { GPURenderTarget } from "./engine/renderers/GPURenderTarget.js"
import { PerspectiveCameraEx } from "./engine/cameras/PerspectiveCameraEx.js"
import { OrbitControls } from "./engine/controls/OrbitControls.js"
import { Color } from "./engine/math/Color.js"
import { ParticleSystem } from "./particleSystem.js"
import { RenderSmoke } from "./render_smoke.js"
import { RenderBlit } from "./render_blit.js"
import { Vector3 } from "./engine/math/Vector3.js"

import { LightTarget } from "./light_target.js"
import { RenderLight } from "./render_light.js"

import { ImageLoader } from "./engine/loaders/ImageLoader.js"
import { RenderFloor } from "./render_floor.js"
import { RenderFloorDepth} from "./render_floor_depth.js"

class Light
{
    constructor()
    {
        this.position = new Vector3(5.0, 5.0, -5.0);    

        this.lightBufferSize = 256;
        this.lightTarget = new LightTarget(this.lightBufferSize,this.lightBufferSize);

        this.lightCamera = new PerspectiveCameraEx(45);
        this.lightCamera.position.set(this.position.x, this.position.y, this.position.z); 
        this.lightCamera.lookAt(0,1,0);
        this.lightCamera.updateMatrixWorld(false);
        this.lightCamera.updateConstant();

    }
}


export function CreateTexture(image, generate_mipmaps = false, srgb = true)
{
    let mipLevelCount = generate_mipmaps? (Math.floor(Math.log2(Math.max(image.width, image.height))) + 1) : 1;        

    let texture = engine_ctx.device.createTexture({
        dimension: '2d',
        size: [image.width, image.height],
        format: srgb? 'rgba8unorm-srgb': 'rgba8unorm',
        mipLevelCount,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });

    let width = image.width;
    let height = image.height;
    let source = image;

    for (let mipLevel =0; mipLevel<mipLevelCount; mipLevel++)
    {
        engine_ctx.device.queue.copyExternalImageToTexture(
            { source },
            { texture, origin: [0, 0], mipLevel},
            [ width, height]
        );

        if (mipLevel < mipLevelCount-1)
        {
            if (width > 1) width = Math.floor(width/2);
            if (height > 1) height = Math.floor(height/2);
            let canvas =  document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            let ctx2d = canvas.getContext("2d");
            ctx2d.drawImage(source, 0,0,width,height);
            source = canvas;
        }        
    }

    return texture;
}

export async function test()
{
    const canvas = document.getElementById('gfx');
    canvas.style.cssText = "position:absolute; width: 100%; height: 100%;";  
    
    const engine_ctx = new EngineContext();
    const canvas_ctx = new CanvasContext(canvas);
    await canvas_ctx.initialize();

    let resized = false;
    const size_changed = ()=>{
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;        
        resized = true;
    };
    
    let observer = new ResizeObserver(size_changed);
    observer.observe(canvas);
    
    let render_target = new GPURenderTarget(canvas_ctx, false);    
    let smoke_target = new GPURenderTarget(null, false);    
    let bind_group_frame  = null;

    let sampler = engine_ctx.device.createSampler({
        addressModeU:"repeat", 
        addressModeV: "repeat",
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: "linear"           
    });

    engine_ctx.cache.bindGroupLayouts.frame = engine_ctx.device.createBindGroupLayout({
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

    const NUM_PARTICLES = 65536;
    
    let psystem = new ParticleSystem(NUM_PARTICLES);
    psystem.reset();
    
    let camera = new PerspectiveCameraEx();
    camera.position.set(0, 1, 5); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(0,1,0); 
    controls.enableDamping = true; 

    let light = new Light();

    let imgLoader = new ImageLoader();
    let floorImage = await imgLoader.loadFile("./assets/floortile.png");
    let floorTex = CreateTexture(floorImage, true);    

    engine_ctx.cache.bindGroupLayouts.render_floor = engine_ctx.device.createBindGroupLayout({
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

    let group_entries_floor = [
        {
            binding: 0,
            resource: floorTex.createView()
        },
        {
            binding: 1,
            resource: sampler
        },
    ];
    
    let bind_group_floor = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.render_floor, entries: group_entries_floor});    
    
    const render = () =>
    {
        controls.update();
        if (resized)
        {
            camera.aspect = canvas.width/canvas.height;
            camera.updateProjectionMatrix();
            resized = false;
        }

        render_target.update();

        let uuid0 = smoke_target.uuid;
        smoke_target.update(render_target.width, render_target.height);    
        if (smoke_target.uuid!=uuid0)
        {
            let group_entries = [
                {
                    binding: 0,
                    resource: smoke_target.view_video
                },
                {
                    binding: 1,
                    resource: sampler
                },
            ];
            
            bind_group_frame = engine_ctx.device.createBindGroup({ layout: engine_ctx.cache.bindGroupLayouts.frame, entries: group_entries});
        }

        camera.updateMatrixWorld(false);
    	camera.updateConstant();

        psystem.update();

        let eyePos = new Vector3(camera.matrixWorld.elements[12], camera.matrixWorld.elements[13], camera.matrixWorld.elements[14]);
        let viewDir = eyePos.clone();
        viewDir.negate();
        viewDir.normalize();
        
        let lightDir = light.position.clone();
        lightDir.negate();
        lightDir.normalize();

        let invertedView = viewDir.dot(lightDir)>0.0;
        let halfVector = lightDir.clone();
        if (invertedView)
        {
            halfVector.add(viewDir);            
        }
        else
        {
            halfVector.sub(viewDir);
        }
        halfVector.normalize();
        psystem.sort(halfVector);
        
        let commandEncoder = engine_ctx.device.createCommandEncoder();

        let depthAttachment = {
            view: render_target.view_depth,
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        };

        // floor depth
        {
            let renderPassDesc = {   
                colorAttachments: [],         
                depthStencilAttachment: depthAttachment 
            }; 
            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

            passEncoder.setViewport(
                0,
                0,
                render_target.width,
                render_target.height,
                0,
                1
            );
        
            passEncoder.setScissorRect(
                0,
                0,
                render_target.width,
                render_target.height,
            );

            RenderFloorDepth(passEncoder, camera, render_target);

            passEncoder.end();

            depthAttachment.depthLoadOp= "load";

        }

        // smoke/light
        {
            let colorAttachment =  {            
                view: smoke_target.view_video,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store'
            };            

            let renderPassDesc = {
                colorAttachments: [colorAttachment],      
                depthStencilAttachment: depthAttachment                  
            }; 

            let colorAttachment_light =  {            
                view: light.lightTarget.view_alpha,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }; 

            let renderPassDesc_light = {
                colorAttachments: [colorAttachment_light],                       
            }; 

            {

                let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);
                passEncoder.setViewport(
                    0,
                    0,
                    smoke_target.width,
                    smoke_target.height,
                    0,
                    1
                );
            
                passEncoder.setScissorRect(
                    0,
                    0,
                    smoke_target.width,
                    smoke_target.height,
                );                

                passEncoder.end();

                colorAttachment.loadOp = "load";

            }

            {
                let passEncoder = commandEncoder.beginRenderPass(renderPassDesc_light);

                passEncoder.setViewport(
                    0,
                    0,
                    light.lightTarget.width,
                    light.lightTarget.height,
                    0,
                    1
                );
            
                passEncoder.setScissorRect(
                    0,
                    0,
                    light.lightTarget.width,
                    light.lightTarget.height,
                );                

                passEncoder.end();

                colorAttachment_light.loadOp = "load";


            }

            let numSlices = 64;
            let batchSize = psystem.numParticles/numSlices;

            for (let i=0; i<numSlices; i++)            
            {           
                // smoke                
                { 
                    let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

                    passEncoder.setViewport(
                        0,
                        0,
                        smoke_target.width,
                        smoke_target.height,
                        0,
                        1
                    );
                
                    passEncoder.setScissorRect(
                        0,
                        0,
                        smoke_target.width,
                        smoke_target.height,
                    );

                    RenderSmoke(passEncoder, camera, light, psystem, smoke_target, batchSize * i, batchSize, invertedView);

                    passEncoder.end();
                }

                // light
                {
                    let passEncoder = commandEncoder.beginRenderPass(renderPassDesc_light);

                    passEncoder.setViewport(
                        0,
                        0,
                        light.lightTarget.width,
                        light.lightTarget.height,
                        0,
                        1
                    );
                
                    passEncoder.setScissorRect(
                        0,
                        0,
                        light.lightTarget.width,
                        light.lightTarget.height,
                    );

                    RenderLight(passEncoder, light.lightCamera, psystem, batchSize * i, batchSize);

                    passEncoder.end();                   
                }
            }
        }

        // floor
        {
            let clearColor = new Color(0.0, 0.0, 0.0);
            let colorAttachment =  {            
                view: render_target.view_video,
                clearValue: { r: clearColor.r, g: clearColor.g, b: clearColor.b, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            };
            

            let renderPassDesc = {
                colorAttachments: [colorAttachment],
                depthStencilAttachment: depthAttachment 
            }; 
            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

            passEncoder.setViewport(
                0,
                0,
                render_target.width,
                render_target.height,
                0,
                1
            );
        
            passEncoder.setScissorRect(
                0,
                0,
                render_target.width,
                render_target.height,
            );

            RenderFloor(passEncoder, camera, light, bind_group_floor, render_target);


            passEncoder.end();
        }

      
        // blit
        {       
            let colorAttachment =  {            
                view: render_target.view_video,                
                loadOp: 'load',
                storeOp: 'store'
            };

            let renderPassDesc = {
                colorAttachments: [colorAttachment]
            }; 
            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

            passEncoder.setViewport(
                0,
                0,
                render_target.width,
                render_target.height,
                0,
                1
            );
        
            passEncoder.setScissorRect(
                0,
                0,
                render_target.width,
                render_target.height,
            );

            RenderBlit(passEncoder, bind_group_frame, render_target);            
          
            passEncoder.end();


        }

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);     
        
        requestAnimationFrame(render);
    }

    render();
}