import torch
import math
import mitsuba as mi
import drjit as dr
import numpy as np
from mitsuba.scalar_rgb import Transform4f as T

class RendererMitsuba:

    def __init__(self, samples, bounces, device, screenWidth, screenHeight):
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.counter = 0
        self.seed_inc = 0
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        mi.set_variant('cuda_ad_rgb')
        mi.LogLevel(1)
        dr.set_device(0)
        self.scene = self.buildInitialScene() # init my scene
        

    def buildInitialScene(self):
        # Create scene
        self.scene = mi.load_dict({
            'type': 'scene',
            'integrator': {
                # 'type': 'aov',
                # 'aovs': 'dd.y:depth',
                # 'my_image':{
                    'type': 'prb'
                    # 'type': 'prb_reparam' 
                    # 'type': 'direct_reparam' #supposed to be better for visibility discontinuities
                # }
            },
            'sensor':  {
                'type': 'perspective',
                'fov': 5,
                'to_world': T.look_at(
                            origin=(0, 0, 0),
                            target=(0, 0, 1),
                            up=(0., -1, 0.)
                        ),
                'film': {
                    'type': 'hdrfilm',
                    'width':  self.screenWidth,
                    'height': self.screenHeight,
                    'pixel_format':'rgba',
                    'sample_border':True,
                },
            },
            "mesh":{
                "type": "obj",
                "filename": "./output/mesh0.obj",
                "face_normals": True,
                'bsdf': {
                    'type': 'principled',
                    'base_color': {
                        'type': 'bitmap',
                        'filename': "./output/diffuseMap_0.png"
                    },                
                    'roughness':{
                        'type': 'bitmap',
                        'filename': "./output/roughnessMap_0.png"
                        
                    }
                }
            },
            'light': {
                'type': 'envmap',
                'filename':'./output/envMap_0.png'
            }
        })         
        # enable grad
        # params = mi.traverse(self.scene)
        # # Mark the green wall color parameter as differentiable
        # dr.enable_grad(params["mesh.vertex_positions"])
        # dr.enable_grad(params["mesh.faces"])
        # dr.enable_grad(params["mesh.vertex_positions"])
        # dr.enable_grad(params["mesh.vertex_normals"])
        # dr.enable_grad(params["mesh.vertex_texcoords"])
        # dr.enable_grad(params["light.data"])
        # # add more here if needed ...
        # # Propagate this change to the scene internal state
        # params.update();
        return self.scene
    
    
    def render(self, vertices, indices, normal, uv, diffuseTexture, specularTexture, roughnessTexture, focal, envMap):
        """take inputs and give it to wrapped function for rendering

        Args:
            vertices (_type_): _description_
            indices (_type_): _description_
            normal (_type_): _description_
            uv (_type_): _description_
            diffuseTexture (_type_): _description_
            specularTexture (_type_): _description_
            roughnessTexture (_type_): _description_
            focal (_type_): _description_
            envMap (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.counter += 1
        self.fov =  torch.tensor([360.0 * torch.atan(self.screenWidth / (2.0 * focal)) / torch.pi]) # from renderer.py
        
        img, grad_img =  RendererMitsuba.render_torch_djit(self.scene, vertices.squeeze(0), indices.to(torch.float32), normal.squeeze(0), uv, diffuseTexture.squeeze(0), specularTexture.squeeze(0), roughnessTexture.squeeze(0), self.fov.item(), envMap.squeeze(0), seed=self.seed_inc) # returns a pytorch
        self.seed_inc += 2 # Make sure to use different seeds
        
        return img
        # return mi.render(scene, scene_params, spp=256, seed=1, seed_grad=2) # return TensorXf
    
    # STANDALONE because of wrap_ad
    @dr.wrap_ad(source='torch', target='drjit')
    def render_torch_djit(scene, vertices, indices, normal, uv, diffuseTexture, specularTexture, roughnessTexture, fov, envMap, spp=8, seed=1):
        """take a texture, update the scene and render it. uses a wrap ad for backpropagation and for gradients
        we are adding a mitsuba computations in a pytorch pipeline

        Returns:
           image : tensorX
        """
        params = mi.traverse(scene)
        params["sensor.x_fov"] = fov
        # update mesh params
        # # -> [1 N 3] -> [N 3]
        params["mesh.vertex_positions"] = dr.ravel(mi.TensorXf(vertices))
        # REVIEW could slow down the whole thing
        params["mesh.faces"] = dr.ravel(mi.TensorXf(indices))
        params["mesh.vertex_normals"] = dr.ravel(mi.TensorXf(normal))
        params["mesh.vertex_texcoords"] = dr.ravel(mi.TensorXf(uv))
        # reflance data is [ X Y 3] so we convert our diffuseTexture to it 
        # update BSDF
        # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-diffuse-material-diffuse
        params["mesh.bsdf.base_color.data"] = mi.TensorXf(diffuseTexture)
        # params["mesh.bsdf.specular.data"] = mi.TensorXf(specularTexture) # principled doesnt support specular textures
        params["mesh.bsdf.roughness.data"] = mi.TensorXf(roughnessTexture)
        
        #update envMaps
        params["light.data"] = mi.TensorXf(envMap)
        #make them differentiable again ?
        # dr.enable_grad(params["mesh.vertex_positions"])
        # dr.enable_grad(params["mesh.faces"])
        # dr.enable_grad(params["mesh.vertex_positions"])
        # dr.enable_grad(params["mesh.vertex_normals"])
        # dr.enable_grad(params["mesh.vertex_texcoords"])
        # dr.enable_grad(params["light.data"])
        
        params.update() 
        img = mi.render(scene, params, spp=8, seed=seed, seed_grad=seed+1)
        grad_img = dr.grad(img)
        # grad_img_light = dr.grad(params["light.data"])
        # # grad_img = dr.grad(params["mesh.vertex_positions"])
        # # see gradients ?
        # from matplotlib import pyplot as plt
        # import matplotlib.cm as cm
        # plt.imshow(grad_img * 2.0)
        # plt.axis('off');
        
        # vlim = dr.max(dr.abs(grad_img))[0]
        # print(f'Remapping colors within range: [{-vlim:.2f}, {vlim:.2f}]')

        # fig, axx = plt.subplots(1, 3, figsize=(8, 3))
        # for i, ax in enumerate(axx):
        #     ax.imshow(grad_img[..., i], cmap=cm.coolwarm, vmin=-vlim, vmax=vlim)
        #     ax.set_title('RGB'[i] + ' gradients')
        #     ax.axis('off')
        # fig.tight_layout()
        # plt.show()  
        
        return img, grad_img
        
    
        

    