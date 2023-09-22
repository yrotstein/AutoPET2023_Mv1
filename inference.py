import numpy as np
import torch 
import torchio as tio 
import torch.nn.functional as F
from UNET3D_v1 import UNet3D_Mirror

import math
from scipy.ndimage.filters import gaussian_filter

################################################################################
        
def inference(ckpt_path, subject, config):
    
    net = UNet3D_Mirror()
    
    patch_size = config['patch_size']
    stride = config['stride']
    bs_test = config['bs_test']
    n_classes = config['n_classes']
    thres_prob = config['thres_prob']
    device = torch.device(config['device'])
    
    model_state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(model_state['model_state_dict'])

    net.to(device)
    net.eval()
    
    mapa_gaussiano = get_gaussian((patch_size,patch_size,patch_size), sigma_scale=1. / 8)
    original_shape = list(subject.shape[1:])
    
    prob_avg = torch.zeros(original_shape)
        
    for nf in range(4):
        
        if nf == 0:
            transforms = tio.Compose([
                tio.Lambda(lambda x: x+1024, types_to_apply=[tio.INTENSITY], include = 'CT'),
                pad_for_patch(patch_size=patch_size, stride=stride),
                tio.Lambda(lambda x: x-1024, types_to_apply=[tio.INTENSITY], include = 'CT'),
                ])
        
        else:
            transforms = tio.Compose([
                tio.Lambda(lambda x: x+1024, types_to_apply=[tio.INTENSITY], include = 'CT'),
                pad_for_patch(patch_size=patch_size, stride=stride),
                tio.Lambda(lambda x: x-1024, types_to_apply=[tio.INTENSITY], include = 'CT'),
                tio.Flip(axes=(nf-1),p=1),
                ])

        transformed = transforms(subject)
        patches,shape_padded = get_patches(transformed,patch_size,stride)
        
        img = torch.cat((patches['CT'].unsqueeze(1),patches['PT'].unsqueeze(1)),dim=1)
        
        ############################################################
        # Perform inference
        output = predict( net, img, bs_test, n_classes, device)

        ############################################################
        # Render complete image from patches
        output = img_render_stride_std(original_shape, shape_padded, output, 
                                       w=mapa_gaussiano, patch_size = patch_size, 
                                       stride=stride, channels=n_classes)
        output = F.softmax(output, dim=0)
        transformed.add_image(tio.ScalarImage(tensor=output, affine=transformed.PT.affine), 'output' )
        
        if nf > 0:
            flip_inv = transformed.get_inverse_transform(warn=False)[0]
            transformed = flip_inv(transformed)
            
        inv = transformed.get_inverse_transform(warn=False)[-1]
        transformed = inv(transformed)
        
        prob_avg = prob_avg + transformed['output'].data.cpu()
    
    prob_avg = prob_avg/4
    
    segm = ( prob_avg[1,:,:,:] > thres_prob ).to(torch.uint8)
        
    return segm.unsqueeze(0)


def predict(net, norm_inputs, bs,n_classes, device):

    segm_outputs_t = torch.zeros((norm_inputs.shape[0],n_classes,norm_inputs.shape[2],norm_inputs.shape[3],norm_inputs.shape[4])).to(device=device)
    n_batch = np.uint16(np.ceil(norm_inputs.shape[0] / bs))
    
    with torch.no_grad():
        for i in range(n_batch):
            si = np.uint16(i*bs)
            ei = np.uint16(i*bs+ bs)
            t_inputs2 = norm_inputs[si:ei,:,:,:,:]
            t_inputs2 = torch.FloatTensor(t_inputs2).to(device)
            t_outputs = net(t_inputs2)
            segm_outputs_t[si:ei,:,:,:,:] = t_outputs
    
    return segm_outputs_t.detach().cpu()



################################################################################
def get_patches( subject, patch_size = 32, stride = 16):
    
    [in_channels,a,b,c] = subject.shape
    
    img_name = ['CT','PT']
        
    ny = int(math.ceil( (a-patch_size)/stride))
    nx = int(math.ceil( (b-patch_size)/stride))  
    nz = int(math.ceil( (c-patch_size)/stride))  
   
    pad0 = ( ny*stride + patch_size ) - a
    pad1 = ( nx*stride + patch_size ) - b
    pad2 = ( nz*stride + patch_size ) - c
    
    assert pad0==0
    assert pad1==0
    assert pad2==0

    n_blocks = (nx+1)*(ny+1)*(nz+1)
    patches={}
    for name in img_name:
        patches[name] = torch.zeros((n_blocks,patch_size,patch_size,patch_size))
    
    sub_img={ k: None for k in img_name }
    
    ii = 0
    for j in range(0,ny+1):
        for i in range(0,nx+1):
            for k in range(0,nz+1):
               
                j_start = j * stride
                j_end = j_start + patch_size
                i_start = i * stride
                i_end = i_start + patch_size
                k_start = k * stride
                k_end = k_start + patch_size
                
                for name in (img_name):
                    sub_img[name] = (subject[name].data.squeeze(0))[j_start:j_end, i_start:i_end , k_start:k_end]

                for name in (img_name):
                    patches[name][ii,:,:,:] = sub_img[name]
                ii = ii +1
                
    for name in (img_name):
        patches[name] = patches[name][:ii,:,:,:]
        
    assert n_blocks == ii 
    
    return patches, [a,b,c]

################################################################################

class pad_for_patch:

    def __init__(self, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, subject):
        _, a, b, c = subject.shape
        
        patch_size = self.patch_size
        stride = self.stride
        
        nx = int(np.ceil( (a - patch_size )/stride))
        ny = int(np.ceil( (b - patch_size )/stride))
        nz = int(np.ceil( (c - patch_size )/stride))
       
        targ_x = nx*stride + patch_size
        targ_y = ny*stride + patch_size
        targ_z = nz*stride + patch_size
        
        pad = tio.CropOrPad(target_shape=(targ_x,targ_y,targ_z))
        padded = pad(subject)

        return padded

################################################################################    
    
def img_render_stride_std(original_shape, new_shape, patches, w, patch_size = 32, stride=16, channels=2):
    
    [a,b,c] = original_shape
    [aa,bb,cc] = new_shape 
    img = torch.zeros((channels,aa,bb,cc)).to(w.device)
    acum = torch.zeros((aa,bb,cc)).to(w.device)
    
    ny = int((aa-patch_size)/stride) + 1
    nx = int((bb-patch_size)/stride) + 1
    nz = int((cc-patch_size)/stride) + 1    
    
    ii = 0
    
    for j in range(0,ny):
        for i in range(0,nx):
            for k in range(0,nz):
                
                j_start = j * stride
                j_end = j_start + patch_size 
                i_start = i * stride
                i_end = i_start + patch_size 
                k_start = k * stride
                k_end = k_start + patch_size 
       
                acum[j_start:j_end, i_start:i_end , k_start:k_end ] = acum[j_start:j_end, i_start:i_end , k_start:k_end ] + w

                for ch in range(channels):
                    sub = patches[ii,ch,:,:,:] 
                    subw = sub * w
                    img[ch,j_start:j_end, i_start:i_end , k_start:k_end ] = img[ch,j_start:j_end, i_start:i_end , k_start:k_end ] + subw
                    
                ii = ii+1

    img = torch.div(img,(acum+1e-9))
    
    return img


def get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])
    
    return torch.from_numpy(gaussian_importance_map)
