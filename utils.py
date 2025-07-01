# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 09:24:14 2025

@author: estumpe
"""


import os
import glob
import numpy as np
import imageio
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
import open3d as o3d
import math
import pickle

class Multimodal3DRegData:
    """Handles loading and organizing multimodal image data from a dataset directory."""
    
    def __init__(self, dataset_dir,modalities):
        """
        Initializes the Multimodal3DRegData object.
    
        Args:
            dataset_dir (str): Path to the dataset root directory.
            modalities (list): List of modality names (e.g., ["infra", "rgb"]).
        """
        
        self.dataset_dir    = dataset_dir
        self.modalities     = modalities  
        self.data_dict      = self.create_data_dict(self.dataset_dir,self.modalities)
        
        
    def create_data_dict(self,dataset_dir, modalities):
        """
        Constructs a dictionary mapping each modality to its file paths and type.
    
        Args:
            dataset_dir (str): Root directory of the dataset.
            modalities (list): List of modalities to load.
    
        Returns:
            dict: Dictionary with modality keys containing image paths and file types.
        """
        
        file_types = [".png",".jpg",".npy"]
        data_dict = {}
        
        for m in modalities:
            
            data_dict[m] = {}
            
            directory = os.path.join(dataset_dir,m)
            
            for f in file_types:
                files = sorted(glob.glob(os.path.join(directory,f"*{f}")))
                if len(files)!=0: 
                    data_dict[m]["paths"]     = files
                    data_dict[m]["file_type"] = f
                    
                    break
            
            if "paths" not in data_dict[m]:
                raise FileNotFoundError(
                    f"No files with supported types ({', '.join(file_types)}) found in {directory}"
                )
        return data_dict
            
    def __getitem__(self, idx):
        """
        Loads and returns image data for all modalities at the specified index.
    
        Args:
            idx (int): Index of the data sample.
    
        Returns:
            dict: Dictionary of images and metadata for each modality.
        """     
        
        img_dict = {}
        
        for m in self.modalities:
            
            img_dict[m] = {}
            
            if self.data_dict[m]["file_type"] in [".png",".jpg"]:
               img_dict[m]["img"] = imageio.imread(self.data_dict[m]["paths"][idx])
            
            elif self.data_dict[m]["file_type"] in [".npy"]:
                img_dict[m]["img"] = np.load(self.data_dict[m]["paths"][idx])
               
            img_dict[m]["path"] = self.data_dict[m]["paths"][idx]
            img_dict[m]["name"] = os.path.basename(img_dict[m]["path"])
                
        return img_dict  
      
    def remap_keys_to_modalities(self, new_modalities): 
        """
        Remaps internal modality keys to new modality names.
    
        Args:
            new_modalities (list): New list of modality names to assign.
        """
        
        remapped = {}
        name_to_modality = dict(zip(self.modalities, new_modalities))
        for m in self.modalities:
            modality = name_to_modality[m]
            remapped[modality] = self.data_dict[m]
        self.data_dict = remapped
        self.modalities = new_modalities





def corner_finder(img,m,checkx,checky,settings,criteria):
    """
    Detects chessboard corners in an image using the specified calibration mode.

    Args:
        img (np.ndarray): Input image (grayscale or RGB).
        m (str): Modality name (used to index the settings).
        settings (dict): Dictionary with calibration settings per modality.
        criteria (tuple): Subpixel refinement termination criteria.

    Returns:
        tuple:
            - ret (bool): Whether corners were successfully found.
            - corners (np.ndarray or None): Detected corner coordinates.
    """
    
    if settings[m]["calib_mode"] == "normal":
        ret, corners  = cv2.findChessboardCorners(img, (checkx,checky),settings[m]["flags"])
        
        if settings[m]["subpix"]!=None and ret:
            cv2.cornerSubPix(img,corners,settings[m]["subpix"],(-1,-1),criteria)
     
    elif settings[m]["calib_mode"] == "SB":
        ret, corners    = cv2.findChessboardCornersSB(img, (checkx,checky),settings[m]["flags"])
        
        if settings[m]["subpix"]!=None and ret:
            cv2.cornerSubPix(img,corners,settings[m]["subpix"],(-1,-1),criteria)
        if ret:
            corners         = corners[::-1] # enforce same corner order as with normal mode
    
    return ret,corners
    
   
def calibrate_modalities(base_modality,supp_modalities,data,checkx,checky,settings,visualize):
    """
    Performs intrinsic and stereo calibration between a base modality and multiple support modalities.

    Args:
        base_modality (str): The reference modality (e.g., "infra").
        supp_modalities (list): List of other modalities to calibrate against the base.
        data (Multimodal3DRegData): Dataset handler instance.
        checkx (int): Number of inner corners along the width of the calibration checkerboard.
        checky (int): Number of inner corners along the height of the calibration checkerboard.
        settings (dict): Dictionary of per-modality calibration parameters.
        visualize (bool): If True, visualizes detected corners during calibration.

    Returns:
        dict: Calibration dictionary mapping each modality to its intrinsic and extrinsic parameters.
    """    
     
    # Termination criteria for refining the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((checkx*checky,3), np.float32)
    objp[:,:2] = np.mgrid[0:checkx,0:checky].T.reshape(-1,2)
    
    print(f"loaded base modality images: {base_modality}")
    
    calibration_dict = {k:{} for k in [base_modality] + supp_modalities}
      
    data.modalities     = [base_modality] #set modalities to base to only load these images
    bm_images           = [img[base_modality]["img"] for img in data]
    
    bm = base_modality
    
    for i,sm in enumerate(supp_modalities):
        
        counter = 0
        
        data.modalities     = [sm] # adjusted to load only the current support modality
        print(f"now loading support modality images: {sm}")
        
        # start of intrinsic calibration
        obj_pts_list   = []
        bm_corner_list = []
        sm_corner_list = [] 
        
        for j,bm_img in enumerate(bm_images):
            
            sm_img = data[j][sm]["img"]
            if sm_img.ndim == 3:
                sm_img = cv2.cvtColor(sm_img, cv2.COLOR_RGB2GRAY)
                        
            ret_bm,corners_bm = corner_finder(bm_img,bm,checkx,checky,settings,criteria)
            
            ret_sm,corners_sm = corner_finder(sm_img,sm,checkx,checky,settings,criteria)
                   
            if ret_bm and ret_sm:
                
                counter+=1
                            
                obj_pts_list.append(objp)
                bm_corner_list.append(corners_bm)
                sm_corner_list.append(corners_sm)
                
            if (j+1)%10==0:
                print(f"extracted {j+1} of {len(bm_images)} corner images ")       
                
            if visualize: 

                bm_img_rgb = np.stack((bm_img,) * 3, axis=-1)
                sm_img_rgb = np.stack((sm_img,) * 3, axis=-1)
                
                if ret_bm:
                    bm_img_rgb = cv2.drawChessboardCorners(bm_img_rgb,(checkx,checky),corners_bm,ret_bm)
                
                if ret_sm:

                    sm_img_rgb = np.ascontiguousarray(sm_img_rgb)
                    sm_img_rgb = cv2.drawChessboardCorners(sm_img_rgb,(checkx,checky),corners_sm,ret_sm)
                
                # Create subplots
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    
                bm_title = "corners found" if ret_bm else "nothing found!"
                sm_title = "corners found" if ret_sm else "nothing found!"
                
                # Plot the images
                axs[0].imshow(bm_img_rgb)
                axs[0].axis('off')  # Hide axes
                axs[0].set_title(bm_title)
    
                axs[1].imshow(sm_img_rgb)
                axs[1].axis('off')
                axs[1].set_title(sm_title)
    
                # Show the figure
                plt.tight_layout()
                plt.show()
                
        print("corner detection finished")
        print(f"detected corners in {counter} of {len(bm_images)} pairs")
        ret_bm, mtx_bm, dist_bm, rvecs_bm, tvecs_bm,std_in_bm,std_ex_bm,per_view_error_bm = cv2.calibrateCameraExtended(obj_pts_list, bm_corner_list, bm_img.shape[::-1],None,None,flags = 0)
        ret_sm, mtx_sm, dist_sm, rvecs_sm, tvecs_sm,std_in_sm,std_ex_sm,per_view_error_sm = cv2.calibrateCameraExtended(obj_pts_list, sm_corner_list, sm_img.shape[::-1],None,None,flags = 0)
        
        print("intrinsic calibration finished")
        
        ret_S, new_mtx_bm, dist_bm, new_mtx_sm, dist_sm, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts_list,
                                                                  bm_corner_list,
                                                                  sm_corner_list,
                                                                  mtx_bm,
                                                                  dist_bm,
                                                                  mtx_sm,
                                                                  dist_sm,
                                                                  bm_img.shape[::-1],
                                                                  criteria_stereo)
    
        print("extrinsic calibration finished")
        
        print(f"intrinsic calibration error for base modality {base_modality}: {ret_bm}")
        print(f"intrinsic calibration error for support modality {sm}: {ret_sm}")
        print(f"extrinsic calibration error: {ret_S}")
        print("\n")
                 
        # the parameters of the base modality are just needed once
        if i == 0:
            
            # intrinsic parameters
            calibration_dict[bm]["mtx"]  = mtx_bm
            calibration_dict[bm]["dist"] = dist_bm
            
            # extrinsic parameters (here the base camera is related to itself, so default values)
            calibration_dict[bm]["Rot"]  = np.eye(3)
            calibration_dict[bm]["Trns"] = np.zeros((3,1))
            calibration_dict[bm]["Trafo"]= np.eye(4)
            calibration_dict[bm]["Emat"] = np.zeros((3,3))
            calibration_dict[bm]["Fmat"] = np.zeros((3,3))
            
        # calibration parameters support modality
        # intrinsic parameters
        calibration_dict[sm]["mtx"]  = mtx_sm
        calibration_dict[sm]["dist"] = dist_sm
        
        # extrinsic parameters (here the base camera is related to itself, so default values)
        calibration_dict[sm]["Rot"]  = Rot
        calibration_dict[sm]["Trns"] = Trns
        T = np.eye(4)
        T[0:3,0:3]  = Rot
        T[0:3,3]    = Trns[:,0]
        
    
        calibration_dict[sm]["Trafo"] = T 
        calibration_dict[sm]["Emat"] = Emat
        calibration_dict[sm]["Fmat"] = Fmat        
       
    # Replace 'old_key' with 'new_key'
    if "hs_rgb" in calibration_dict:
        calibration_dict["hs"] = calibration_dict.pop("hs_rgb")
        
    return calibration_dict


# Depth related functions

def pixel_coord_np(width, height):
    """
    Computes homogeneous pixel coordinates.

    Args:
        width (int): Image width.
        height (int): Image height.

    Returns:
        np.ndarray: Array of shape (3, H*W) representing (x, y, 1) coordinates.
    """
    
    x = np.linspace(0, width - 1, width).astype(np.int32)
    y = np.linspace(0, height - 1, height).astype(np.int32)
    [x, y] = np.meshgrid(x, y)
    
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

def project_depth_to_pcl(depth_img,trafo):
    """
    Projects depth image into a point cloud using camera intrinsics.

    Args:
        depth_img (np.ndarray): Depth image.
        trafo (dict): Calibration data with 'mtx' and 'dist' keys.

    Returns:
        tuple: (point cloud as Nx3 array, depth pixel indices as Nx2 array).
    """
    
    depth_indices       = pixel_coord_np(depth_img.shape[1], depth_img.shape[0]).T
    depth_values        = depth_img[depth_indices[:,1],depth_indices[:,0]]
       
    depth_uv_und        = depth_indices.copy()
    depth_uv_und[:,0:2] = cv2.undistortPointsIter(np.expand_dims(depth_uv_und[:,0:2].astype(np.float64),0),trafo["mtx"],trafo["dist"],R = None,P = trafo["mtx"],criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))[:,0,:]
    
    point_vector        = np.dot(np.linalg.inv(trafo["mtx"]),depth_uv_und.T).T
    pcl                 = point_vector*np.expand_dims(depth_values,-1)
        
    return pcl,depth_indices


def extract_pcl_ROI(pcl,b_dict):
    """
    Applies ROI filtering to the point cloud based on 3D bounds.

    Args:
        pcl (np.ndarray): Point cloud (N, 3).
        b_dict (dict): Dictionary with 'xmin', 'xmax', etc. bounds.

    Returns:
        np.ndarray: Filtered point cloud with out-of-bounds points set to [0,0,0].
    """     
    #here we create boolean arrays to indicate which xyz points lie within the set boundaries
    x_bound = (np.logical_and(pcl[:,0]>=b_dict["xmin"], pcl[:,0]<=b_dict["xmax"])) # get all points which are within the bounding box xmin and xmax
    y_bound = (np.logical_and(pcl[:,1]>=b_dict["ymin"], pcl[:,1]<=b_dict["ymax"])) # get all points which are within the bounding box ymin and ymax
    z_bound = (np.logical_and(pcl[:,2]>=b_dict["zmin"], pcl[:,2]<=b_dict["zmax"])) # in the pointcloud array get all points which are within the specified width d --> list with True for these points

    res_bound = z_bound*x_bound*y_bound #intersection of all boundary operations
    
    pcl[~res_bound]= np.array([0,0,0]) #set all values which are outside of the boundaries to [0,0,0] --> these points are then skipped in the pcl_to_mesh function
      
    return pcl


def pcl_to_mesh(pcl, h, w, minAngle=15):
    """
    Converts point cloud into a triangle mesh using pixel adjacency.

    Args:
        pcl (np.ndarray): Point cloud (N, 3).
        h (int): Image height.
        w (int): Image width.
        minAngle (float): Minimum surface angle to accept triangle (avoids flying pixels).

    Returns:
        tuple: (Open3D mesh, Open3D raycasting scene)
    """
    
    # each point corresponds to a coordinate in the depth map (camera) established before via pixel_coord_np in project_depth_to_pcl
    # we can therefore iterate height and width over pcl/cam_coords
    
    cam_coords      = pcl.T # each point corresponds to a coordinate in the depth map (camera) established before via pixel_coord_np in project_depth_to_pcl
    
    indices = o3d.utility.Vector3iVector()

    for i in range(0, h-1):
        for j in range(0, w-1):
            verts = [
                cam_coords[:, w*i+j],     # going by height and with the corresponding depthmap[j,i] corresponds to the xyz information in cam_coords[:, w*i+j]
                cam_coords[:, w*(i+1)+j], # we also take the neighboring pixels to create the triangle meshes
                cam_coords[:, w*i+(j+1)],
            ]
            if [0,0,0] in map(list, verts): # skip all instances, where value was set to [0,0,0] when establishing the boundary restrictions
                pass
            else:
                v1 = verts[0] - verts[1] # to create the mesh we create vertices and coresponding node indices
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle: # we can enforce that only vertices within the min angle are used (e.g. to prevent flying pixel artifacts)
                    indices.append([w*i+j, w*(i+1)+j, w*i+(j+1)])
        
            # to create a dense mesh we also need to add triangles for the other pixel neighbors
            verts = [
                cam_coords[:, w*i+(j+1)],
                cam_coords[:, w*(i+1)+j],
                cam_coords[:, w*(i+1)+(j+1)],
            ]
            if [0,0,0] in map(list, verts):
                pass
            else:
                v1 = verts[0] - verts[1]
                v2 = verts[0] - verts[2]
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                center = (verts[0] + verts[1] + verts[2]) / 3.0
                u = center / np.linalg.norm(center)
                angle = math.degrees(math.asin(abs(np.dot(n, u))))
                if angle > minAngle:
                    indices.append([w*i+(j+1),w*(i+1)+j, w*(i+1)+(j+1)])
                
    points = o3d.utility.Vector3dVector(cam_coords.transpose())

    mesh = o3d.geometry.TriangleMesh(points, indices)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()   

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    
    return mesh,scene

def compute_uncertainty_mesh(pcl,border_edges_np,b_dict):
    """
    Creates uncertainty mesh based on mesh boundary edges.

    Args:
        pcl (np.ndarray): Point cloud.
        border_edges_np (np.ndarray): Boundary edge indices.
        b_dict (dict): Bounding box dictionary for mesh extent.

    Returns:
        tuple: (uncertainty mesh, raycasting scene, point index dictionary).
    """    
    
    pi_dict         = {} # point index dictionary
    points_list     = []
    vert_list = o3d.utility.Vector3iVector()
    
    for i in range(border_edges_np.shape[0]): # we loop over all boundary edges (each consisting of 2 indices that link to the points in pcl that form the edge)
        for j in range(2): # iterating over both of the 2 indices
            ind = border_edges_np[i,j] # get the corresponding pcl point index
            if(str(ind) in pi_dict): # skip, if this index was already used (since we are using edges, these points will always appear twice for neighboring edges)
                pass
            else:
                coords              = pcl[ind] # get the 3D point
                ground_coords       = (coords/coords[2])*b_dict["zmax"] # project that point onto the ground plane and save new location
                curr_index          = len(points_list) #record the position in saving both coords and ground coords in points_list
                
                pi_dict[str(ind)]                   = {}
                pi_dict[str(ind)]["object"]         = coords
                pi_dict[str(ind)]["ground"]         = ground_coords
                pi_dict[str(ind)]["object_index"]   = curr_index
                pi_dict[str(ind)]["ground_index"]   = curr_index+1
                
                points_list.append(coords)
                points_list.append(ground_coords)
                
        # create vert indices
        # Now, after looping over both ende of an edge we have recorded 4 positions (2* object, 2*ground) --> they form a parallelogram, 
        # which we now represent as two triangles (vertsL,vertsR)
        vertsL = [
            pi_dict[str(border_edges_np[i,0])]["object_index"],
            pi_dict[str(border_edges_np[i,1])]["object_index"],
            pi_dict[str(border_edges_np[i,0])]["ground_index"]
        ]
        
        
        vertsR = [
            pi_dict[str(border_edges_np[i,0])]["ground_index"],
            pi_dict[str(border_edges_np[i,1])]["ground_index"],
            pi_dict[str(border_edges_np[i,1])]["object_index"]
        ]
        
        vert_list.append(vertsL)
        vert_list.append(vertsR)
       
    points_arr = np.asarray(points_list)     
            
    points = o3d.utility.Vector3dVector(points_arr)
    
    mesh = o3d.geometry.TriangleMesh(points, vert_list)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()              
    mesh.paint_uniform_color([1, 0.706, 0])   
     
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    return mesh,scene,pi_dict 


# ingoing ray related functions
 
def generate_rays(img,trafo):
    """
    Generates normalized ray directions for each pixel in an image.

    Args:
        img (np.ndarray): Input image.
        trafo (dict): Intrinsic calibration matrix under key 'mtx'.

    Returns:
        tuple: (ray directions Nx4, pixel coordinates Nx2).
    """
    
    uv         = pixel_coord_np(img.shape[1], img.shape[0]).T # get all the pixel u,v coordinate positions of the input image
    ray_vector = np.dot(np.linalg.inv(trafo["mtx"]),uv.T).T # using the intrinsic matrix, we compute the ray directions for each pixel position uv
    norm       = np.expand_dims(np.linalg.norm(ray_vector,axis = 1),axis=-1) 
    ray_vector = ray_vector/norm # we normalize the rays to a length of 1
    ray_vector = np.c_[ray_vector,np.ones(ray_vector.shape[0])] # we add an array column filled with ones (necessary to do computations with 4x4 transformation matrices)
    
    return ray_vector, uv

def ray_intersection(ray_vector,trafo,scene,u_scene):
    """
    Casts rays and checks intersection with object and uncertainty meshes.

    Args:
        ray_vector (np.ndarray): Ray directions (N, 4).
        trafo (dict): Contains transformation matrix.
        scene (o3d.t.geometry.RaycastingScene): Object mesh scene.
        u_scene (o3d.t.geometry.RaycastingScene): Uncertainty mesh scene.

    Returns:
        tuple: (object mesh intersection distances, uncertainty mesh distances).
    """

    # define the origin of each ray, which is in [0,0,0] of the cameras coordinate systems
    origin          = np.c_[np.zeros((ray_vector.shape[0],3)),np.ones(ray_vector.shape[0])]
    
    # we now transform the origin into the base coordinate system (in our case corresponding to the infrared/depth) for easier computation
    origin_k0       = np.dot(np.linalg.inv(trafo["Trafo"]),origin.T).T
    origin_k0       = np.add(origin_k0,0.000001) # add something for slight offset to prevent rays from casting directly onto mesh nodes, which causes visual artifacts
    
    # Next we do the same with the ray vector, we computed from before
    ray_vector_k0   = np.dot(np.linalg.inv(trafo["Trafo"]),ray_vector.T).T
    ray_vector_k0   = ray_vector_k0 - origin_k0 # the o3d cast scene requires the difference vector between rays and the origin
    
    rays            = np.c_[origin_k0[:,0:3],ray_vector_k0[:,0:3]].astype(np.float32) # concatenate both
    
    # next we compute wether the cast rays hit the two meshes
    # if a ray does not hit, np.inf is the result
    # for instances where a ray hits the mesh, the value represents the distance (our rays have base length = 1 from normalizing, the value is therefore: 1*distance)
    # note the distance represents the normed distance using the defined scale (1 = length of checkerboard square width)
    scene_hit       = (scene.cast_rays(rays))["t_hit"].numpy() 
    u_scene_hit     = (u_scene.cast_rays(rays))["t_hit"].numpy()       
    
    return scene_hit, u_scene_hit

def compute_uncertainty(scene_hit,u_scene_hit):
    """
    Determines uncertain pixels based on ray intersection distances.

    Args:
        scene_hit (np.ndarray): Distances for object mesh intersections.
        u_scene_hit (np.ndarray): Distances for uncertainty mesh intersections.

    Returns:
        tuple: (boolean mask for valid object hits, mask for uncertain pixels).
    """
    
    ray_diff = u_scene_hit - scene_hit # computes the difference between lengths between the ray hitting the object and the uncertatinty mesh

    # the following are all boolean arrays, which can then be used to filter pixel coordinates for masking
    obj_bool_arr = (scene_hit!=np.inf)           # marks all rays (corresponding pixel indices) that hit the object mesh. rays that don't hit have np.inf as length 
    um_bool_arr  = (u_scene_hit!=np.inf)         # marks all rays (corresponding pixel indices) that hit the uncertainty mesh. rays that don't hit have np.inf as length 
    um_bool_closer = (np.sign(ray_diff)==-1)   # marks all rays where the uncertainty mesh was hit before the object mesh, does not include np.inf consideration yet
    
    uncertain_bool_arr = (obj_bool_arr*um_bool_arr*um_bool_closer) # uncertainty case: 1. ray hits object (obj_bool_arr) AND 2. ray also hits uncerctainty mesh (um_bool_arr) AND 3. uncertainty mesh is hit at closer distance (um_bool_closer)
        
    return obj_bool_arr,uncertain_bool_arr

    
def compute_3D_points_ks0(ray_vector,trafo,scene_hit,obj_bool_arr):
    """
    Calculates 3D points in the base coordinate frame where rays intersect the object mesh.

    Args:
        ray_vector (np.ndarray): Ray vectors (N, 4).
        trafo (dict): Calibration and transformation info.
        scene_hit (np.ndarray): Distances for mesh intersection.
        obj_bool_arr (np.ndarray): Boolean mask for valid intersections.

    Returns:
        np.ndarray: 3D points in base frame.
    """
    
    xyz_ksx         = ray_vector.copy()
    xyz_ksx[:,0:3]  = np.multiply(ray_vector[:,0:3],np.expand_dims(scene_hit,-1)) # we multiply the ray vectors (each has length = 1) with the distance
    
    xyz_ks0         = np.dot(np.linalg.inv(trafo["Trafo"]),xyz_ksx.T).T #transform the points into the base coordinate system
    xyz_ks0         = xyz_ks0[obj_bool_arr] # use the boolean array to only take the points that have hit the object
    
    return xyz_ks0


def target_3D_registration(target_modality,target_img,calib_dict,scene,u_scene):
    """
    Computes ray intersections from target image to meshes and extracts 3D data.

    Args:
        target_modality (str): Name of the target modality.
        target_img (np.ndarray): Target image.
        calib_dict (dict): Calibration dictionary for all modalities.
        scene (Open3D scene): Object mesh scene.
        u_scene (Open3D scene): Uncertainty mesh scene.

    Returns:
        dict: Contains 3D points, UV coords, and uncertainty mask for target.
    """    

    # for each pixel position we create rays with help from the intrinsic camera matrix
    target_ray_vector,target_uv = generate_rays(target_img,calib_dict[target_modality])
    
    # next we compute the ray intersections with the meshes (object mesh, uncertainty mesh)
    target_scene_hit, target_u_scene_hit = ray_intersection(target_ray_vector,calib_dict[target_modality],scene,u_scene)
        
    # now, we find out whether uncertainty or object mesh is hit first to define the ingoing uncertainty error
    obj_bool_arr,uncertain_bool_arr = compute_uncertainty(target_scene_hit,target_u_scene_hit) #um_hit_closer
    
    # using the information from the cast rays, we compute the intersection points of the cast rays with the object mesh
    target_xyz_ks0  = compute_3D_points_ks0(target_ray_vector,calib_dict[target_modality],target_scene_hit,obj_bool_arr)
    
    reg_in = {}
    reg_in["xyz"]          = target_xyz_ks0 # the 3D intersection points with the mesh used to cast the outgoing rays
    reg_in["obj_uv"]       = target_uv[obj_bool_arr] # the u,v coordinates of the target image that hit the obj mesh
    reg_in["uncertain_uv"] = target_uv[uncertain_bool_arr] # the u,v coordinates of the target image that hit the uncertainty mesh first
    reg_in["temp"] = obj_bool_arr
    return reg_in


# outgoing ray related functions

def ray_backcast(xyz_ks0,trafo,scene,u_scene):
    """
    Back-projects rays from 3D points to source camera and checks mesh intersections.

    Args:
        xyz_ks0 (np.ndarray): 3D points in base coordinate system.
        trafo (dict): Calibration and transformation matrix.
        scene (Open3D scene): Object mesh.
        u_scene (Open3D scene): Uncertainty mesh.

    Returns:
        tuple: (object mesh hit distances, uncertainty mesh hit distances)
    """
        
    origin      = np.c_[np.zeros((xyz_ks0.shape[0],3)),np.ones(xyz_ks0.shape[0])]
     
    #transform origin into the base coordinate system
    origin_k0   = np.dot(np.linalg.inv(trafo["Trafo"]),origin.T).T
     
    # this part here below constitutes the difference to the ray_intersection function
    # here we directly take the 3D intersection positions (xyz_ks0) to construct the rays
    # the rays in this case are also not unit vectors (length = 1), but have the length that directly corresponds to 
    # the distance of the source camera and intersection points
    # this helps to identify occlusions easily later, since scene_hit will mostly contain values that are 1 (ray hit distance = distance between source/intersection points)
    # but values smaller than 1 then indicate occlusion cases
    rays        = np.c_[origin_k0[:,0:3], xyz_ks0[:,0:3]-origin_k0[:,0:3]].astype(np.float32)   
      
    scene_hit   = (scene.cast_rays(rays))["t_hit"].numpy()
    u_scene_hit = (u_scene.cast_rays(rays))["t_hit"].numpy()
        
    return scene_hit,u_scene_hit


def compute_img_coords(xyz_ks0,img,trafo,uv_target):
    """
    Projects 3D points into image plane and filters out-of-bounds projections.

    Args:
        xyz_ks0 (np.ndarray): 3D points in base coordinate system.
        img (np.ndarray): Source image.
        trafo (dict): Source modality calibration.
        uv_target (np.ndarray): UV coordinates in target image.

    Returns:
        tuple: (filtered UV in target, 3D points, UV in source)
    """
    
    # we transform the intersection xyz points into the source coordinate system (projection)
    xyz_ks_source = np.dot(trafo["Trafo"],xyz_ks0.T).T
    
    # using the intrinsic calibration information to compute the corresponding uv coordinates
    uv_source   = np.round(np.dot(trafo["mtx"],xyz_ks_source.T[0:3,:]/xyz_ks_source.T[2,:]).T).astype(np.int32)
    
    # some of the xyz points might not visible to the source camera (outside of field of view)
    # we therefore want to filter out all u,v coordinates that lie outside the source image boundaries
    
    h_bound     = (np.logical_and(uv_source[:,1]>=0,
                              uv_source[:,1]<img.shape[0]))
                
    w_bound     = (np.logical_and(uv_source[:,0]>=0,
                              uv_source[:,0]<img.shape[1]))
    
    b_mask          = h_bound*w_bound #using the boolean array b_mask we filter out invalid coordinates and only keep legitimate ones
    b_mask_indices  = np.argwhere(b_mask)[:,0] # we find the indices where b_mask is True
    
    # we now apply these indices to the source,target, and 3D coordinates to establish the the relationship between registered coordinates
    uv_source_out   = uv_source[b_mask_indices] 
    uv_target_out   = uv_target[b_mask_indices]
    xyz_out         = xyz_ks0[b_mask_indices]
     
    return uv_target_out,xyz_out,uv_source_out


def target_source_registration(target,source,source_img,calib_dict,scene,u_scene,reg_in):
    """
    Registers a source modality to the target via 3D mesh projection.

    Args:
        target (str): Target modality name.
        source (str): Source modality name.
        source_img (np.ndarray): Source image.
        calib_dict (dict): Calibration dictionary.
        scene (Open3D scene): Object mesh.
        u_scene (Open3D scene): Uncertainty mesh.
        reg_in (dict): Intermediate data from target registration.

    Returns:
        dict: Registered UVs, uncertainty and occlusion masks, 3D points/colors.
    """
    
    # cast the rays back from the source views onto the object mesh and uncertainty mesh
    source_scene_hit,source_u_scene_hit = ray_backcast(reg_in["xyz"],calib_dict[source],scene,u_scene)

    # we now compute the registered coordinates between source and target + the matching 3D points
    uv_target_out,xyz_out,uv_source_out = compute_img_coords(reg_in["xyz"],source_img,calib_dict[source],reg_in["obj_uv"])
     
    # again, we identify the rays that hit the uncertainty mask first (from the source camera perspective)
    obj_bool_arr,uncertain_bool_arr = compute_uncertainty(source_scene_hit,source_u_scene_hit)
    # in addition we compute, which pixels are occluded by finding those that hit the mesh at a closer distance (<=0.99)
    occlusion_bool_arr = np.logical_and(source_scene_hit<=0.99,source_scene_hit!=np.inf)         
        
    reg_out = {}
    reg_out["uv_target"] = uv_target_out
    reg_out["xyz"]       = xyz_out
    reg_out["uv_source"] = uv_source_out
    reg_out["uncertain_uv"] = reg_in["obj_uv"][uncertain_bool_arr] # pixel positions in target that are uncertain
    reg_out["occlusion_uv"] = reg_in["obj_uv"][occlusion_bool_arr] # pixel positions in target that are occluded 
    
    return reg_out


def convert_3D_thermal_colors_to_cmap(temperature,cmap,vmin):
    """
    Converts thermal scalar values to RGB colormap.

    Args:
        temperature (np.ndarray): Temperature values.
        cmap (str): Name of the matplotlib colormap.
        vmin (float): Minimum value for normalization.

    Returns:
        np.ndarray: RGB colors as uint8 array.
    """
    
    vmax = temperature.max()
    
    # we need to normalize values to 0,1 to save as pointcloud
    norm = (temperature - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)
    
    # Apply colormap
    colormap = cm.get_cmap(cmap)
    colors_rgba = colormap(norm)  # shape: (N, 4), includes alpha
    
    # Drop the alpha channel
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
    
    return colors_rgb
    

def save_point_cloud(points,colors,filename):
    """
    Saves a point cloud with colors to a PLY file.

    Args:
        points (np.ndarray): 3D coordinates (N, 3).
        colors (np.ndarray): RGB or grayscale values (N, 3) or (N,).
        filename (str): Output path for the PLY file.
    """
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors.ndim == 1 or colors.shape[1] == 1:
        # Normalize and map grayscale to RGB (repeat across channels)
        gray = colors.reshape(-1, 1)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)  # normalize to [0, 1]
        rgb = np.repeat(gray, 3, axis=1)
    
    elif colors.shape[1] == 3:
        rgb = colors/255.  # ensure RGB is in [0, 1]

    elif colors.shape[1] == 204:
        
        print(f" only for demonstration purposes, the hyperspec image with 204 channels is saved as 3D information from 50th channel")
        spectrum = colors[:,49:50] # take 50th channel
        
        spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
                                                    
        rgb = np.repeat(spectrum, 3, axis=1)                                           
         
    else: 
        raise ValueError("Colors must be either shape (N,) or (N, 3)")

    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Save to file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


