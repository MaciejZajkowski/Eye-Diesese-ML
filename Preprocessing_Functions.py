import cv2
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image





def get_edges_stats(img,img_name) -> dict:
    """returns stats (0 normalized count,endge picels mean and median)

    Args:
        img (np.array): has to be grayscaled 2 dim image
        img_name (str): name in returned dictionary
    """
    
    def calculate_stats(arr,type):
        
        ret_dict={}
        
        count = pd.Series(arr).value_counts(normalize=True)
        
        if 0 not in count.index:
            ret_dict[img_name+ '_' + type + '_edge_0_norm_count'] = 0
        else:
            ret_dict[img_name + '_' + type + '_edge_0_norm_count'] =  count.loc[0]
            
        ret_dict[img_name+'_' + type + '_edge_median'] = np.median(arr)
        ret_dict[img_name+ '_' + type + '_edge_mean'] = arr.mean()
            
        return ret_dict
            
            
        
    img_shape =img.shape
    
    final_ret = {}
    
    edges = np.concatenate((img[[0,img_shape[0]-1],:].ravel(),img[1:-1,[0,img_shape[1]-1]].ravel()))
    
    final_ret.update(calculate_stats(edges,'all'))
    final_ret.update(calculate_stats(img[:,-1],'right'))
    final_ret.update(calculate_stats(img[-1,:],'bottom'))
    final_ret.update(calculate_stats(img[0,:],'up'))
    final_ret.update(calculate_stats(img[0,:],'left'))
    
    
    return final_ret


def preprocess_one_img(path:str,save_path:str):
    filename = path.split('/')[-1]
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = cnts[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    
    f_savepath = save_path + '/'+filename
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) 
    crop = cv2.resize(crop, (224, 224,3))
    cv2.imwrite(f_savepath,crop)
    
    return_dict ={'save_path':f_savepath}
    
    return_dict.update(get_edges_stats(gray,'org'))
    
    
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return_dict.update(get_edges_stats(crop_gray,'crop'))
    
    return return_dict

def get_stats_from_paths(paths_table):
    
    final_table = []
    
    for path in tqdm(paths_table):
        
        if path.split('/')[-2] == 'compressed':
            name = 'crop'
        else:
            name = 'org'
        try:
            im = np.array(Image.open(path).convert('L'))
            result = get_edges_stats(im,name)
            result.update({'X_'+name: im.shape[0],'Y_'+name:im.shape[1]})
            result.update({'error':False})
        except:
            result = {'error':True}

        final_table.append(result)
        
    return pd.DataFrame(final_table)