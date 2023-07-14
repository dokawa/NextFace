import math
from sphericalharmonics import SphericalHarmonics
from morphablemodel import MorphableModel
from renderer import Renderer
from camera import Camera
from customRenderer import *
from utils import *
from image import saveImage
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import polyscope as ps
class Pipeline:

    def __init__(self,  config):
        '''
        a pipeline can generate and render textured faces under different camera angles and lighting conditions
        :param config: configuration file used to parameterize the pipeline
        '''
        self.config = config
        self.device = config.device
        self.camera = Camera(self.device)
        self.sh = SphericalHarmonics(config.envMapRes, self.device)

        if self.config.lamdmarksDetectorType == 'fan':
            pathLandmarksAssociation = '/landmark_62.txt'
        elif self.config.lamdmarksDetectorType == 'mediapipe':
            pathLandmarksAssociation = '/landmark_62_mp.txt'
        else:
            raise ValueError(f'lamdmarksDetectorType must be one of [mediapipe, fan] but was {self.config.lamdmarksDetectorType}')

        self.morphableModel = MorphableModel(path = config.path,
                                             textureResolution= config.textureResolution,
                                             trimPca= config.trimPca,
                                             landmarksPathName=pathLandmarksAssociation,
                                             device = self.device
                                             )
        self.renderer = Renderer(config.rtTrainingSamples, 1, self.device)
        self.uvMap = self.morphableModel.uvMap.clone()
        self.uvMap[:, 1] = 1.0 - self.uvMap[:, 1]
        self.faces32 = self.morphableModel.faces.to(torch.int32).contiguous()
        self.shBands = config.bands
        self.sharedIdentity = False

    def initSceneParameters(self, n, sharedIdentity = False):
        '''
        init pipeline parameters (face shape, albedo, exp coeffs, light and  head pose (camera))
        :param n: the the number of parameters (if negative than the pipeline variables are not allocated)
        :param sharedIdentity: if true, the shape and albedo coeffs are equal to 1, as they belong to the same person identity
        :return:
        '''

        if n <= 0:
            return

        self.sharedIdentity = sharedIdentity
        nShape = 1 if sharedIdentity == True else n

        self.vShapeCoeff = torch.zeros([nShape, self.morphableModel.shapeBasisSize], dtype = torch.float32, device = self.device)
        self.vAlbedoCoeff = torch.zeros([nShape, self.morphableModel.albedoBasisSize], dtype=torch.float32, device=self.device)

        self.vExpCoeff = torch.zeros([n, self.morphableModel.expBasisSize], dtype=torch.float32, device=self.device)
        self.vRotation = torch.zeros([n, 3], dtype=torch.float32, device=self.device)
        self.vTranslation = torch.zeros([n, 3], dtype=torch.float32, device=self.device)
        self.vTranslation[:, 2] = 500.
        self.vRotation[:, 0] = 3.14
        self.vFocals = self.config.camFocalLength * torch.ones([n], dtype=torch.float32, device=self.device)
        self.vShCoeffs = 0.0 * torch.ones([n, self.shBands * self.shBands, 3], dtype=torch.float32, device=self.device)
        self.vShCoeffs[..., 0, 0] = 0.5
        self.vShCoeffs[..., 2, 0] = -0.5
        self.vShCoeffs[..., 1] = self.vShCoeffs[..., 0]
        self.vShCoeffs[..., 2] = self.vShCoeffs[..., 0]

        texRes = self.morphableModel.getTextureResolution()
        self.vRoughness = 0.4 * torch.ones([nShape, texRes, texRes, 1], dtype=torch.float32, device=self.device)

    def computeShape(self):
        '''
        compute shape vertices from the shape and expression coefficients
        :return: tensor of 3d vertices [n, verticesNumber, 3]
        '''

        assert(self.vShapeCoeff is not None and self.vExpCoeff is not None)
        vertices = self.morphableModel.computeShape(self.vShapeCoeff, self.vExpCoeff)
        return vertices

    def transformVertices(self, vertices = None):
        '''
        transform vertices to camera coordinate space
        :param vertices: tensor of 3d vertices [n, verticesNumber, 3]
        :return:  transformed  vertices [n, verticesNumber, 3]
        '''

        if vertices is None:
            vertices = self.computeShape()

        assert(vertices.dim() == 3 and vertices.shape[-1] == 3)
        assert(self.vTranslation is not None and self.vRotation is not None)
        assert(vertices.shape[0] == self.vTranslation.shape[0] == self.vRotation.shape[0])

        transformedVertices = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)
        return transformedVertices

    def render(self, cameraVerts = None, diffuseTextures = None, specularTextures = None, roughnessTextures = None, renderAlbedo = False, vertexBased = False):
        '''
        ray trace an image given camera vertices and corresponding textures
        :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
        :param diffuseTextures: diffuse textures tensor [n, texRes, texRes, 3]
        :param specularTextures: specular textures tensor [n, texRes, texRes, 3]
        :param roughnessTextures: roughness textures tensor [n, texRes, texRes, 1]
        :param renderAlbedo: if True render albedo else ray trace image
        :param vertexBased: if True we render by vertex instead of ray tracing
        :return: ray traced images [n, resX, resY, 4]
        '''
        
        if cameraVerts is None :
            vertices, diffAlbedo, specAlbedo = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
            cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)

        #compute normals
        normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)

        if diffuseTextures is None:
            diffuseTextures = self.morphableModel.generateTextureFromAlbedo(diffAlbedo)

        if specularTextures is None:
            specularTextures = self.morphableModel.generateTextureFromAlbedo(specAlbedo)

        if roughnessTextures is None:
            roughnessTextures  = self.vRoughness

        envMaps = self.sh.toEnvMap(self.vShCoeffs)

        assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
        assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
        assert (diffuseTextures.dim() == 4 and diffuseTextures.shape[1] == diffuseTextures.shape[2] == self.morphableModel.getTextureResolution() and diffuseTextures.shape[-1] == 3)
        assert (specularTextures.dim() == 4 and specularTextures.shape[1] == specularTextures.shape[2] == self.morphableModel.getTextureResolution() and specularTextures.shape[-1] == 3)
        assert (roughnessTextures.dim() == 4 and roughnessTextures.shape[1] == roughnessTextures.shape[2] == self.morphableModel.getTextureResolution() and roughnessTextures.shape[-1] == 1)
        assert(cameraVerts.shape[0] == envMaps.shape[0])
        assert (diffuseTextures.shape[0] == specularTextures.shape[0] == roughnessTextures.shape[0])

        if vertexBased:
            # do we take in account rotation when computing normals ?
            # coeff -> face shape -> face shape rotated -> face vertex (based on camera) -> normals (based on face vertex) -> rotated normals *
            diffAlbedo = self.morphableModel.computeDiffuseAlbedo(self.vAlbedoCoeff)
            specAlbedo = self.morphableModel.computeSpecularAlbedo(self.vAlbedoCoeff)
            
            vertexColors = self.computeVertexColor(diffAlbedo, specAlbedo, roughnessTextures, normals)
            # face_shape -> self.computeShape() -> vertices (if no camera) or cameraVerts (if  camera)
            images = self.computeVertexImage(cameraVerts, vertexColors, debug=False)
        else:
            scenes = self.renderer.buildScenes(cameraVerts, self.faces32, normals, self.uvMap, diffuseTextures, specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0), self.vFocals, envMaps)
            if renderAlbedo:
                images = self.renderer.renderAlbedo(scenes)
            else:
                images = self.renderer.render(scenes)
                
        return images
    def renderVertexBased(self, cameraVerts = None, diffuseAlbedo = None, specularAlbedo = None, albedoOnly= False, lightingOnly=False ):
        '''
        render the vertices in an image given camera vertices and corresponding albedos
        :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
        :param diffuseAlbedo: diffuse textures tensor [n, texRes, texRes, 3]
        :param specularAlbedo: specular textures tensor [n, texRes, texRes, 3]
        :return: ray traced images [n, resX, resY, 4]
        '''
        if cameraVerts is None or diffuseAlbedo is None:
            vertices, diffuseAlbedo, specularAlbedo = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
            cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)
        #compute normals
        normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)
        assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
        
        vertexColors = self.computeVertexColor(diffuseAlbedo, specularAlbedo, normals,albedoOnly=albedoOnly,lightingOnly=lightingOnly)
        # face_shape -> self.computeShape() -> vertices (if no camera) or cameraVerts (if  camera)
        images = self.computeVertexImage(cameraVerts, vertexColors, debug=False)
                
        return images
    
    def landmarkLoss(self, cameraVertices, landmarks, focals, cameraCenters,  debugDir = None):
        '''
        calculate scalar loss between vertices in camera space and 2d landmarks pixels
        :param cameraVertices: 3d vertices [n, nVertices, 3]
        :param landmarks: 2d corresponding pixels [n, nVertices, 2]
        :param landmarks: camera focals [n]
        :param cameraCenters: camera centers [n, 2
        :param debugDir: if not none save landmarks and vertices to an image file
        :return: scalar loss (float)
        '''
        assert (cameraVertices.dim() == 3 and cameraVertices.shape[-1] == 3)
        assert (focals.dim() == 1)
        assert(cameraCenters.dim() == 2 and cameraCenters.shape[-1] == 2)
        assert (landmarks.dim() == 3 and landmarks.shape[-1] == 2)
        assert cameraVertices.shape[0] == landmarks.shape[0] == focals.shape[0] == cameraCenters.shape[0]

        headPoints = cameraVertices[:, self.morphableModel.landmarksAssociation]
        assert (landmarks.shape[-2] == headPoints.shape[-2])

        projPoints = focals.view(-1, 1, 1) * headPoints[..., :2] / headPoints[..., 2:]
        projPoints += cameraCenters.unsqueeze(1)
        loss = torch.norm(projPoints - landmarks, 2, dim=-1).pow(2).mean()
        if debugDir:
            for i in range(projPoints.shape[0]):
                image = saveLandmarksVerticesProjections(self.inputImage.tensor[i], projPoints[i], self.landmarks[i])
                cv2.imwrite(debugDir + '/lp' +  str(i) +'.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return loss
    
    # Generate colors for each vertices
    def computeVertexColor(self, diffAlbedo, specAlbedo, normals, roughnessTexture = None,  gamma = None, albedoOnly=False, lightingOnly=False):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            diffAlbedo     -- torch.tensor, size (B, N, 3) 
            normals        -- torch.tensor, size (B, N, 3), rotated face normal
            Y              -- torch.tensor, size (B, N, 81), sh basis functions, should be (order +1 )^2
        """
        if albedoOnly :
            return diffAlbedo

        gammaInit = 2.2
        # vShCoeffs is of shape [1, 81, 3]
        sh = self.vShCoeffs
        Y = self.sh.preComputeSHBasisFunction(normals,sh_order=8)
        r = Y @ sh[..., :1]
        g = Y @ sh[..., 1:2]
        b = Y @ sh[..., 2:]
        if lightingOnly:
            face_color = torch.cat([r, g, b], dim=-1)
        else:
            face_color = diffAlbedo * torch.cat([r, g, b], dim=-1)
            face_color = torch.clamp(face_color, min=1e-8)

        return face_color
    # predict face and mask
    def computeVertexImage(self, cameraVertices, verticesColor, debug=False) : 
        #since we already have the cameraVertices
        width = 256
        height = 256
        fov = torch.tensor([360.0 * torch.atan(width / (2.0 * self.vFocals)) / torch.pi]) # from renderer.py
        far = 100
        near = 0.1
        
        # 1- find projectionMatrix based from camera : camera space -> clip space
        projMatrix = self.perspectiveProjMatrix(fov,width/height, near, far).to(self.device)
        # Apply the projection matrix to vertices
        ones = torch.ones(cameraVertices.shape[0], cameraVertices.shape[1], 1).to(self.device)
        homogeneous_vertices = torch.cat((cameraVertices, ones), -1)
        vertices_in_clip_space = (projMatrix @ homogeneous_vertices.transpose(-1, -2)).transpose(-1, -2)

        # Normalize the vertices (divide by W) and put them in cartesian coordinates
        vertices_in_clip_space = vertices_in_clip_space[..., :3] / vertices_in_clip_space[..., 3:]
        # Ignore vertices where x, y,  > 1 or < -1
        mask = torch.abs(vertices_in_clip_space[:, :, :2]) <= 1.0  # Only consider x and y coordinates
        mask = mask.all(dim=-1)  # All x and y must satisfy the condition
        vertices_in_clip_space = vertices_in_clip_space[mask]

        # Convert vertices from clip space to screen space
        vertices_in_screen_space = vertices_in_clip_space.clone()
        vertices_in_screen_space[..., 0] = (vertices_in_clip_space[..., 0] + 1) / 2 * width
        vertices_in_screen_space[..., 1] = (vertices_in_clip_space[..., 1] + 1) / 2 * height

        # Vertices color manipulation
        verticesColor = verticesColor.squeeze(0)#mean(dim=1)# 
        mask = mask.squeeze(0)
        # print("masking")
        # print(verticesColor.shape)
        # print(mask.shape)
        colors_in_screen_space = verticesColor[mask]

        # Create an empty image
        image_data = torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)
        # Convert your vertices and colors to an image
        for i, vertex in enumerate(vertices_in_screen_space):
            x, y = int(vertex[0]), int(vertex[1])
            if 0 <= x < width and 0 <= y < height:
                # Add color to the pixel
                image_data[y, x, :] += colors_in_screen_space[i, :]

        # Add batch dimension to image_data
        image_data = image_data.unsqueeze(0)
        # add alpha channel to rgb
        alpha_channel = torch.ones((1, 256, 256, 1)).to(self.device)  # Ensure it's on the same device as your image
        image_data = torch.cat((image_data, alpha_channel), dim=-1)  # Append alpha channel
        
        if debug:            
            normals = self.morphableModel.meshNormals.computeNormals(cameraVertices)
            self.displayTensorColorAndNormals(vertices_in_screen_space,verticesColor,normals)

        return image_data
    
    def perspectiveProjMatrix(self, fov, aspect_ratio, near, far):
        """
        Create a perspective projection matrix.
        
        :param fov: field of view angle in the y direction (in degrees).
        :param aspect_ratio: aspect ratio of the viewport (width/height).
        :param near: distance to the near clipping plane.
        :param far: distance to the far clipping plane.
        """
        f = 1.0 / torch.tan(torch.deg2rad(fov) / 2.0)
        # right handed matrix
        projMatrix = torch.tensor([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, far/(far-near), -(far*near)/(far-near)],
            [0, 0, 1, 0]
        ])
        # or flip on x
        # projMatrix = torch.tensor([
        #     [-f / aspect_ratio, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, far/(far-near), -(far*near)/(far-near)],
        #     [0, 0, 1, 0]
        # ])
        # or flip on y axis
        # projMatrix = torch.tensor([
        #     [f / aspect_ratio, 0, 0, 0],
        #     [0, -f, 0, 0],
        #     [0, 0, far/(far-near), -(far*near)/(far-near)],
        #     [0, 0, 1, 0]
        # ])
        # or flip on z axis
        # projMatrix = torch.tensor([
        #     [f / aspect_ratio, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, -far/(far-near), (far*near)/(far-near)],
        #     [0, 0, -1, 0]
        # ])
        # left handed
        # projMatrix = torch.tensor([
        #     [f / aspect_ratio, 0, 0, 0],
        #     [0, -f, 0, 0],
        #     [0, 0, -far/(far-near), -(far*near)/(far-near)],
        #     [0, 0, -1, 0]
        # ])
        return projMatrix
    # draw the visuals
    def displayTensor(self, vertices):
        """util function to display tensors

        Args:
            vertices (Tensor of shape [1, N, 3]): vertices to display (in cartesian coordinates)
        """
        # Convert tensor to numpy array
        scatter_np = vertices.detach().cpu().numpy()
        # Reshape scatter_np to [N, 3] for scatter()
        scatter_np = np.squeeze(scatter_np, axis=0) 
        # Compute the mask
        mask = np.abs(scatter_np[:, :2]) <= 1.0  # Only consider x and y coordinates
        mask = mask.all(axis=-1)  # All x and y must satisfy the condition

        # Create a color array (all 'green' initially)
        colors = np.array(['green'] * len(mask), dtype=str)

        # Change the color of vertices NOT corresponding to the mask to 'red'
        colors[~mask] = 'red'

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot, with colors based on the mask
        ax.scatter(scatter_np[:, 0], scatter_np[:, 1], scatter_np[:, 2], c=colors)
        # Show axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()  
    def displayTensorInPolyscope(self,vertices):
        """display a point cloud 

        Args:
            vertices (N x 3 tensor): vertices that we want to display
        """
        # Convert tensor to numpy array
        scatter_np = vertices.detach().cpu().numpy()
        # Reshape scatter_np to [N, 3] for scatter()
        scatter_np = np.squeeze(scatter_np, axis=0) 
        ps.init()
        # `my_points` is a Nx3 numpy array
        ps.register_point_cloud("my vertices", scatter_np)
        ps.show()
    def displayTensorColorAndNormals(self, verticesPosition, verticesColor, verticesNormals):
        """
        verticesPosition (N, 3) : vertices to display
        verticesColor (N, 3): color of vertices
        verticesNormals (1, N, 3): normals of each vertices
        
        """  
         # Convert tensor to numpy array
        position = verticesPosition.detach().cpu().numpy()
        colors = verticesColor.detach().cpu().numpy()
        normals = verticesNormals.detach().cpu().numpy()
        # Reshape colors and normals to [N, 3] for scatter()
        normals = np.squeeze(normals, axis=0) 
        
        ps.init()
        # use volume mesh
        ps_cloud = ps.register_point_cloud("points",position, enabled=True)
        ps_cloud.add_color_quantity("colors",colors)
        ps_cloud.add_vector_quantity("normals",normals,enabled=True)
      
        ps.show()   
    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lkm.detach().cpu().numpy()
                output_vis_numpy = self.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = self.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)
    def draw_landmarks(img, landmark, color='r', step=2):
        """
        Return:
            img              -- numpy.array, (B, H, W, 3) img with landmark, RGB order, range (0, 255)
            

        Parameters:
            img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
            landmark         -- numpy.array, (B, 68, 2), y direction is opposite to v direction
            color            -- str, 'r' or 'b' (red or blue)
        """
        if color =='r':
            c = np.array([255., 0, 0])
        else:
            c = np.array([0, 0, 255.])

        _, H, W, _ = img.shape
        img, landmark = img.copy(), landmark.copy()
        landmark[..., 1] = H - 1 - landmark[..., 1]
        landmark = np.round(landmark).astype(np.int32)
        for i in range(landmark.shape[1]):
            x, y = landmark[:, i, 0], landmark[:, i, 1]
            for j in range(-step, step):
                for k in range(-step, step):
                    u = np.clip(x + j, 0, W - 1)
                    v = np.clip(y + k, 0, H - 1)
                    for m in range(landmark.shape[0]):
                        img[m, v[m], u[m]] = c
        return img