import tensorflow as tf
import cv2
import os
import numpy as np
from EFCHandler import EFCHandler
from ellipse_my import ellipse_my

    
root_dataset='/home/gp/repos/data'
label_data_folder='fcn_anno'
image_data_folder='train_data'
image_save_folder='image_save20180519_valid'
file_name=['training','validation']
#file_name=['validation', 'training']
txt_file_name=['validation.txt','training.txt']
extend_rate=1.2
resize_crop_sz=224
part_num=200
coef_num=30
#the round pixel to get
step=3

def _rect(gt):
    
    lx=int(float(gt[1]))
    ly=int(float(gt[2]))
    rx=int(float(gt[3]))
    ry=int(float(gt[4]))
    w=rx-lx
    h=ry-ly
    
    return lx,ly,rx,ry,w,h

def get_point(x,y):
    s_x=int(x-step)
    e_x=int(x+step+step)
    s_y=int(y-step)
    e_y=int(y+step+step)
    
    if step==0:
        e_x=int(e_x+1)
        e_y=int(e_y+1)
    
    return s_x,e_x,s_y,e_y
    

label_folder=os.path.join(root_dataset,label_data_folder)
file_list=[f for f in os.listdir(label_folder) if f.endswith(".txt")]
t=0

for f in file_list:
    
    print('txt_name',f)
    print('filename',file_name[t])
    print('----------------------------')
    
    records=[]
    f_path=os.path.join(root_dataset,label_data_folder,f)
    gt_file=open(f_path)
    
    gt=[]
    while True:
        line=gt_file.readline()
        if not line:
            break
        line=line.split(' ')
        gt.append(line)
    
    gt_file.close()
    l=len(gt)
    print('The number of data is %d' % l)
    
    folder=os.path.join(root_dataset,image_save_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    img_folder0=os.path.join(folder,'images')
    if not os.path.exists(img_folder0):
        os.mkdir(img_folder0)
    img_folder1=os.path.join(img_folder0,file_name[t])
    if not os.path.exists(img_folder1):
        os.mkdir(img_folder1)
        
    gt_folder0=os.path.join(folder,'annotations')
    if not os.path.exists(gt_folder0):
        os.mkdir(gt_folder0)
    gt_folder1=os.path.join(gt_folder0,file_name[t])
    if not os.path.exists(gt_folder1):
        os.mkdir(gt_folder1)
    
    size=(resize_crop_sz,resize_crop_sz)
    
    
    for i in range(l):
        
        print('gt[i][0]',gt[i][0])
        im = cv2.imread(gt[i][0],cv2.IMREAD_COLOR)
        frame_sz = np.array([len(im),len(im[0])])
        #frame_sz=im.shape
        gt_im=np.zeros((frame_sz[0],frame_sz[1],1))
        #print('img_name',gt[i][0],'frame_sz',frame_sz)
        
        lx,ly,rx,ry,w,h=_rect(gt[i])
        #print('lx:',lx,'ly:',ly,'rx',rx,'ry',ry,'w',w,'h',h)
        cx=lx+w/2
        cy=ly+h/2
       
        
        img_name_split=gt[i][0].split('/')
        str_l=len(img_name_split)
        img_name=os.path.join(img_folder1,img_name_split[str_l-3]+img_name_split[str_l-1])
       
        crops=cv2.resize(im,size,interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_name,crops)
        
        
        box=[cx,cy,h/2,w/2]
        #print('box:',box)
        pts=ellipse_my(box,0)
        for j in range(len(pts)):
            x=int(pts[j][0])
            y=int(pts[j][1])
            s_x,e_x,s_y,e_y=get_point(x,y)
            gt_im[s_y:e_y,s_x:e_x,:]=1
        
        gt_img_name=os.path.join(gt_folder1,img_name_split[str_l-3]+img_name_split[str_l-1])
        gt_crops=cv2.resize(gt_im,size,interpolation=cv2.INTER_AREA)
        #print('gt_crops',gt_crops.shape)
        cv2.imwrite(gt_img_name,gt_crops)
        if i % 1000 == 0:
            print('Now is %d...' % i)
        '''line=img_name+' '+gt_img_name+' '+str(c_x)+' '+str(c_y)+' '+str(n_w)+' '+str(n_h)+' '+str(0)+'\n'
        #print(line)
        records.append(str(line))
    out_file_path=os.path.join(folder,txt_file_name[t])
    out_file = open(out_file_path,'w')
    for record in records:
        print(record)
        out_file.write(record)
    out_file.close()'''
    
    t=t+1
  
