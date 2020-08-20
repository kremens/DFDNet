import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_crop
from util.visualizer import save_images
from util import html
from util import util
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import cv2
import dlib
from skimage import transform as trans
from skimage import io
from data.image_folder import make_dataset
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/FaceLandmarkDetection')

import face_alignment

import shutil

from dfl.dfl_read import load_data


# Chuan: Add this line to avoid CUDNN error
torch.backends.cudnn.enabled = False


def get_part_location(dfl_image, image):
    Landmarks = dfl_image.get_landmarks()

    width, height = image.size
    if width != 512 or height != 512:
        width_scale = 512.0 / width
        height_scale = 512.0 / height

        Landmarks = Landmarks * np.array([width_scale, height_scale])

    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))

    #left eye
    Mean_LE = np.mean(Landmarks[Map_LE],0)
    L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
    Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
    #right eye
    Mean_RE = np.mean(Landmarks[Map_RE],0)
    L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
    Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)

    #nose
    Mean_NO = np.mean(Landmarks[Map_NO],0)
    L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
    Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)

    #mouth
    Mean_MO = np.mean(Landmarks[Map_MO],0)
    L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
    Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)

    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)


def main():
    ###########################################################################
    ################# functions of crop and align face images #################
    ###########################################################################
    def get_5_points(img):
        dets = detector(img, 1)
        if len(dets) == 0:
            return None
        areas = []
        if len(dets) > 1:
            print('\t###### Warning: more than one face is detected. In this version, we only handle the largest one.')
        for i in range(len(dets)):
            area = (dets[i].rect.right()-dets[i].rect.left())*(dets[i].rect.bottom()-dets[i].rect.top())
            areas.append(area)
        ins = areas.index(max(areas))
        shape = sp(img, dets[ins].rect) 
        single_points = []
        for i in range(5):
            single_points.append([shape.part(i).x, shape.part(i).y])
        return np.array(single_points) 

    def align_and_save(img_path, save_path, save_input_path, save_param_path, upsample_scale=2):
        out_size = (512, 512) 
        img = dlib.load_rgb_image(img_path)
        h,w,_ = img.shape
        source = get_5_points(img) 
        if source is None: #
            print('\t################ No face is detected')
            return
        tform = trans.SimilarityTransform()                                                                                                                                                  
        tform.estimate(source, reference)
        M = tform.params[0:2,:]
        crop_img = cv2.warpAffine(img, M, out_size)
        io.imsave(save_path, crop_img) #save the crop and align face
        io.imsave(save_input_path, img) #save the whole input image
        tform2 = trans.SimilarityTransform()  
        tform2.estimate(reference, source*upsample_scale)
        # inv_M = cv2.invertAffineTransform(M)
        np.savetxt(save_param_path, tform2.params[0:2,:],fmt='%.3f') #save the inverse affine parameters
        
    def reverse_align(input_path, face_path, param_path, save_path, upsample_scale=2):
        out_size = (512, 512) 
        input_img = dlib.load_rgb_image(input_path)
        h,w,_ = input_img.shape
        face512 = dlib.load_rgb_image(face_path)
        inv_M = np.loadtxt(param_path)
        inv_crop_img = cv2.warpAffine(face512, inv_M, (w*upsample_scale,h*upsample_scale))
        mask = np.ones((512, 512, 3), dtype=np.float32) #* 255
        inv_mask = cv2.warpAffine(mask, inv_M, (w*upsample_scale,h*upsample_scale))
        upsample_img = cv2.resize(input_img, (w*upsample_scale, h*upsample_scale))
        inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2 * upsample_scale, 2 * upsample_scale), np.uint8))# to remove the black border
        inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
        total_face_area = np.sum(inv_mask_erosion_removeborder)//3
        w_edge = int(total_face_area ** 0.5) // 20 #compute the fusion edge based on the area of face
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center,(blur_size + 1, blur_size + 1),0)
        merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * upsample_img
        io.imsave(save_path, merge_img.astype(np.uint8))

    ###########################################################################
    ################ functions of preparing the test images ###################
    ###########################################################################
    def AddUpSample(img):
        return img.resize((512, 512), Image.BICUBIC)

    def get_part_location(dfl_image, image):
        Landmarks = dfl_image.get_landmarks()

        width, height = image.size
        if width != 512 or height != 512:
            width_scale = 512.0 / width
            height_scale = 512.0 / height

            Landmarks = Landmarks * np.array([width_scale, height_scale])

        Map_LE = list(np.hstack((range(17,22), range(36,42))))
        Map_RE = list(np.hstack((range(22,27), range(42,48))))
        Map_NO = list(range(29,36))
        Map_MO = list(range(48,68))

        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)

        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)

        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)

        return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)

    
    def obtain_inputs(img_path, img_name, Type):
        A_paths = os.path.join(img_path,img_name)
        Imgs = Image.open(A_paths).convert('RGB')
        dfl_image = load_data(A_paths)

        Part_locations = get_part_location(dfl_image, Imgs)
        if Part_locations == 0:
            print('wrong part_location')
            return 0
        width, height = Imgs.size
        L = min(width, height)

        #################################################
        A= Imgs
        C = A
        A = AddUpSample(A)

        A = transforms.ToTensor()(A) 
        C = transforms.ToTensor()(C)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) #
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) #

        return {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': A_paths,'Part_locations': Part_locations}
        

    def obtain_inputs_without_parts(img_path, img_name, Type):
        A_paths = os.path.join(img_path,img_name)
        Imgs = Image.open(A_paths).convert('RGB')


        width, height = Imgs.size
        L = min(width, height)

        #################################################
        A= Imgs
        C = A
        A = AddUpSample(A)

        A = transforms.ToTensor()(A) 
        C = transforms.ToTensor()(C)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) #
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) #

        return {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': A_paths}

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest' #
    opt.enchanced_folder_name = 'enchanced'

    if not os.path.exists(opt.test_dir + '_' + opt.enchanced_folder_name):
        os.makedirs(opt.test_dir + '_' + opt.enchanced_folder_name)

    #######################################################################
    ########################### Test Param ################################
    #######################################################################
    opt.gpu_ids = [opt.gpu_id] # gpu id. if use cpu, set opt.gpu_ids = []
    TestImgPath = opt.test_dir # test image path
    AlignedImgPath = opt.aligned_dir # aligned image path
    ResultsDir = opt.test_dir + '_' + opt.enchanced_folder_name  #save path 

    print('Creating model ... ')
    model = create_model(opt)
    model.setup(opt)
    
    # test
    ImgNames = os.listdir(TestImgPath)
    ImgNames.sort()

    for i, ImgName in enumerate(ImgNames):
        # print(ImgName)
        if not opt.aligned_dir:
            data = obtain_inputs(TestImgPath, ImgName, 'real')
        else:
            data_aligned = obtain_inputs(AlignedImgPath, '.'.join(ImgName.split('.')[:-1]) + opt.aligned_postfix, 'real')
            data = obtain_inputs_without_parts(TestImgPath, ImgName, 'real')
            data['Part_locations'] = data_aligned['Part_locations']
        
        if data == 0:
            print ('Skipping ' + ImgName + ' data not found');
            continue
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        output_path = ResultsDir + '/' + ImgName
        print(output_path)

        for label, image in visuals.items():
            if label == 'fake_A':
                image_numpy = util.tensor2im(image)
                image_pil = Image.fromarray(image_numpy)
                image_pil.save(output_path)


if __name__ == '__main__':
    main()