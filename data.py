from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random
import math
import time
import pickle
from collections import defaultdict

TAG_DIM = 5


def loadfiles():
    ch_grp_cat = pickle.load(open('./Toy_Dataset/all_city/CH/gcat_grp_cat','rb'))
    ny_grp_cat = pickle.load(open('./Toy_Dataset/all_city/NY/gcat_grp_cat','rb'))
    sf_grp_cat = pickle.load(open('./Toy_Dataset/all_city/SF/gcat_grp_cat','rb'))
    return [ch_grp_cat, ny_grp_cat, sf_grp_cat]

def get_metric_ch(grp,city,grp_cat_array):
    ch_grp_cat, ny_grp_cat, sf_grp_cat = grp_cat_array
    if city==0:
        cat_name=ch_grp_cat[grp]
        if cat_name==0 or cat_name==1 or cat_name==4:
            return 1
        if cat_name==2 or cat_name==3:
            return 2
        return 3
    else:
        if city==1:
            cat_name=ny_grp_cat[grp]
            if cat_name==0 or cat_name==1:
                return 1
            return 2
        else:   
            cat_name=sf_grp_cat[grp]
            if cat_name==0 or cat_name==1:
                return 1
            if cat_name==3:
                return 2
            return 2

def get_grp_cat(grp,city,grp_cat_array):
    ch_grp_cat, ny_grp_cat, sf_grp_cat = grp_cat_array
    if city==0:
        cat_name=ch_grp_cat[grp]
    else:
        if city==1:
            cat_name=ny_grp_cat[grp]
        else:   
            cat_name=sf_grp_cat[grp]
    return cat_name     


def read_thres(city, thres1, thres2):
    if city==0:
        s1="CH"
        s2="Chicago"
    else:
        if city==1:
            s1="NY"
            s2="NewYork"
        else:
            s1="SF"
            s2="SanFrancisco"
    
    fname1="./Toy_Dataset/all_city/"+s1+"/"+s2+"_thresholds_last10.txt"
    fp=open(fname1,"r")
    cc=0
    
    for line in fp:
        line=line.rstrip()
        line=line.split()
        thres1[city][cc]={}
        thres2[city][cc]={}
        thres1[city][cc][0]=int(line[2])
        thres2[city][cc][0]=int(line[3])
        thres1[city][cc][1]=int(line[4])
        thres2[city][cc][1]=int(line[5])
        thres1[city][cc][2]=int(line[6])
        thres2[city][cc][2]=int(line[7])
        cc+=1
    fp.close()

    #UPDATE ATTENDANCE AND SIZE THRESHOLDS WITH NEW THRESHOLDS
    fname1="./Toy_Dataset/all_city/"+s1+"/"+s2+"_thres_all_windows.txt"
    fp=open(fname1,"r")
    cc=0
    for line in fp:
        line=line.rstrip()
        line=line.split()
        thres1[city][cc][0]=float(line[0])*0.001
        thres2[city][cc][0]=float(line[1])*0.001
        thres1[city][cc][1]=int(line[2])
        thres2[city][cc][1]=int(line[3])
        cc+=1
    
    #print thres1,thres2

def input_data(city, lab, minval, maxval, thres1, thres2, gs, grp_cat_array):
    print("-input_data- :",city)
    if city==0:
        s1="CH"
    else:
        if city==1:
            s1="NY"
        else:
            s1="SF"
    
    p = pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_x5",'rb'))
    q = pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_tag5",'rb'))
    grp_id = pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_grp5",'rb'))
    window = pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_window5",'rb'))

    group_attendance_growth=pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_growth5",'rb'))
    group_size=pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_size5",'rb'))

    new_group_size = pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_size_raw5",'rb'))
    new_group_attendance_growth = pickle.load(open("./Toy_Dataset/all_city/"+s1+"/all_event_we_att_raw5",'rb'))

    print('-input_data- :',"length of group size" )
    print('-input_data- :',len(group_size))

    print('-input_data- :',"length of 5 raw group size")
    print('-input_data- :',len(new_group_size))

    print('-input_data- :',"lenth of gag")
    print('-input_data- :',len(group_attendance_growth))

    print('-input_data- :',"length of 5 raw gag")
    print('-input_data- :',len(new_group_attendance_growth))

    #print(new_gorup_size)
    # print(group_size)
    # print("gs[0]")
    # print(group_size[0])

    # Normalizing Inputs
    '''
    maxval = float(max(group_size.values()))
    for grp in group_size:
        group_size[grp]=group_size[grp]/maxval

    # print group_attendance_growth.values()
    # exit()
    '''
    minval[city] =0.0
    '''
    minval[city] = abs(min(group_size.values()))
    #minval=0.0
    print 'Minval = ',minval[city]
    for grp in group_size:
        group_size[grp]=group_size[grp]+minval[city]

    #print group_attendance_growth.values()
    '''
    
    # maxval[city] = float(max(group_size.values())) if gs else float(max(group_attendance_growth.values()))

    maxg1 = float(max(group_size.values()))
    maxg2 = float(max([max(d) for d in new_group_size.values()]))
    maxe1 = float(max(group_attendance_growth.values()))
    maxe2 = float(max([max(d) for d in new_group_attendance_growth.values()]))

    maxval[city] = maxg2 if gs else maxe2

    print('-input_data-e1 for citxg1 = ',maxval[city])
    #exit()
    
    for grp in group_size:
        group_size[grp]=group_size[grp]/maxg1

    for grp in new_group_size:
        for i in range(len(new_group_size[grp])):
            new_group_size[grp][i] = new_group_size[grp][i]/maxg2
    
    for grp in group_attendance_growth:
         group_attendance_growth[grp]=group_attendance_growth[grp]/maxe1

    for grp in new_group_attendance_growth:
        for i in range(len(new_group_attendance_growth[grp])):
            new_group_attendance_growth[grp][i] = new_group_attendance_growth[grp][i]/maxe2

    #for grp in group_attendance_growth:
    #    group_attendance_growth[grp]=group_attendance_growth[grp]/10

    print('-input_data- :',min(group_size.values()),max(group_size.values()))
    #exit()

    

    grp_cat_array = loadfiles()
    print('-input_data- :',len(p),len(lab))
    for k in p:
        grp_catt=get_grp_cat(grp_id[k],city, grp_cat_array)
        e_succ_thres1=thres2[city][grp_catt][1]
        e_fail_thres1=thres1[city][grp_catt][1]
        e_succ_thres2=thres2[city][grp_catt][0]
        e_fail_thres2=thres1[city][grp_catt][0]
        if gs:
            
            # if ((get_metric_ch(grp_id[k],city,grp_cat_array) == 2 and (((group_size[k]*maxval[city]) >= e_succ_thres1 and q[k]==1) or ((group_size[k]*maxval[city]) <= e_fail_thres1 and q[k]==0)))): #and (((group_attendance_growth[k]*maxval[city]) >= e_succ_thres2 and q[k]==1) or ((group_attendance_growth[k]*maxval[city]) <= e_fail_thres2 and q[k]==0))): #  get_metric_ch(grp_id[k],city,grp_cat_array) == 2 and
            #     lab.append((grp_id[k],np.array(p[k],dtype=np.float32),q[k],group_size[k],group_size[k],city,new_group_size[k])) #USING ONLY GROUP SIZE

            if (((get_metric_ch(grp_id[k],city,grp_cat_array) == 2) or (get_metric_ch(grp_id[k],city,grp_cat_array) == 1)) and 
                (((group_size[k]*maxg1) >= e_succ_thres1 and q[k]==1) or ((group_size[k]*maxg1) <= e_fail_thres1 and q[k]==0)) or
                (((group_attendance_growth[k]*maxe1) >= e_succ_thres2 and q[k]==1) or ((group_attendance_growth[k]*maxe1) <= e_fail_thres2 and q[k]==0))): #and (((group_attendance_growth[k]*maxval[city]) >= e_succ_thres2 and q[k]==1) or ((group_attendance_growth[k]*maxval[city]) <= e_fail_thres2 and q[k]==0))): #  get_metric_ch(grp_id[k],city,grp_cat_array) == 2 and
                lab.append((grp_id[k],np.array(p[k],dtype=np.float32),q[k],group_size[k],group_size[k],city,new_group_size[k])) #USING ONLY GROUP SIZE
        
        else:
            
            if (((get_metric_ch(grp_id[k],city,grp_cat_array) == 2) or (get_metric_ch(grp_id[k],city,grp_cat_array) == 1)) and 
                (((group_size[k]*maxg1) >= e_succ_thres1 and q[k]==1) or ((group_size[k]*maxg1) <= e_fail_thres1 and q[k]==0)) or
                (((group_attendance_growth[k]*maxe1) >= e_succ_thres2 and q[k]==1) or ((group_attendance_growth[k]*maxe1) <= e_fail_thres2 and q[k]==0))): #and (((group_attendance_growth[k]*maxval[city]) >= e_succ_thres2 and q[k]==1) or ((group_attendance_growth[k]*maxval[city]) <= e_fail_thres2 and q[k]==0))): #  get_metric_ch(grp_id[k],city,grp_cat_array) == 2 and
                lab.append((grp_id[k],np.array(p[k],dtype=np.float32),q[k],group_attendance_growth[k],group_attendance_growth[k],city,new_group_attendance_growth[k])) #USING ONLY GROUP SIZE


def get_thres():
    thres1={}
    thres2={}
    for i in range(0,3):
        thres1[i] = {}
        thres2[i] = {}

    for i in range(3):
        read_thres(i, thres1, thres2)
    
    return thres1, thres2

def getlist(gs):

    thres1={}
    thres2={}
    for i in range(0,3):
        thres1[i]={}
        thres2[i]={}

    read_thres(0, thres1, thres2)
    read_thres(1, thres1, thres2)
    read_thres(2, thres1, thres2)
    lab=[]
    minval={}
    maxval={}
    grp_cat_array = loadfiles()
    input_data(0, lab, minval, maxval, thres1, thres2, gs, grp_cat_array)
    input_data(1, lab, minval, maxval, thres1, thres2, gs, grp_cat_array)
    input_data(2, lab, minval, maxval, thres1, thres2, gs, grp_cat_array)
    print("-getlist- : lab",len(lab))

    #random.shuffle(lab)

    # creating balanced dataset    k refer ground truth value
    lab1=[]
    ctt0=0
    ctt1=0
    for i,j,k,l1,l2,city,l3 in lab:
        if k==0:
            ctt0+=1
        else:
            ctt1+=1

    print("-getlist- : size with k==0",ctt0)
    print("-getlist- : size with k==1",ctt1)
    ct=min(ctt0,ctt1)
    ctt0=0
    ctt1=0
    for i,j,k,l1,l2,city,l3 in lab:
        if k==0:
            ctt0+=1
            if ctt0<=ct:
                lab1.append((i,j,k,l1,l2,city,l3))
        else:
            ctt1+=1
            if ctt1<=ct:
                lab1.append((i,j,k,l1,l2,city,l3))
    lab=lab1
    #random.shuffle(lab)

    '''
    eg_list=[]
    gk_list=[]
    lab1=[]

    for i,j,k,l in lab:
        s = group_size[i]
        g = group_attendance_growth[i]
        lab1.append((i,j,s,g,l))
        if get_metric_ch(i)==1:
            eg_list.append(g)
        gk_list.append(s)


    lab=lab1
    '''
    print("-getlist- : data length ",len(lab))
    #random.shuffle(lab)
    metadata = [minval, maxval, thres1, thres2]
    return lab, metadata
    #exit()

def e_get_sample(lab, length, start):
    feat=[]
    tag=[]
    tar=[]
    grp=[]
    grw=[]
    siz=[]
    cit=[]
    grw5 =[]

    #lab = getlist()
    
    s1=0
    #s2=5*15
    s2=5*19
    #s3=5*15+5*14
    for i in range(start,start+length):
        elem=lab[i][1]
        tag_elem=np.array(elem[:TAG_DIM],dtype=np.float32)
        feat_all=elem[TAG_DIM:]
        #feat_elem=[feat_all[j:j + H_DIMS] for j in xrange(0, len(feat_all), H_DIMS)]
        feat_elem=[]
        
        
        for j in range(0,5):
            '''
            s11=s1+j*15
            feat1=feat_all[s11:s11+15]
            s22=s2+j*14
            feat2=feat_all[s22:s22+14]
            s33=s3+j*18
            feat3=feat_all[s33:s33+18]
                # 3rd one length = 18
            #print len(feat1),len(feat2)
            feat4=np.append(feat1,feat2)
            feat_elem.append(np.append(feat4,feat3))
            '''
            s11=s1+j*19
            feat1=feat_all[s11:s11+19]
            s22=s2+j*14
            feat2=feat_all[s22:s22+14]
            feat_elem.append(np.append(feat1,feat2))
        
        feat_elem=np.array(feat_elem, dtype=np.float32)
        #print feat_elem.shape
        feat.append(feat_elem)
        tag.append(tag_elem)
        tar_elem=np.array([lab[i][2]],dtype=np.float32)
        tar.append(tar_elem)
        grp.append(lab[i][0])
        
        grw_elem=np.array([lab[i][3]],dtype=np.float32)
        grw.append(grw_elem)
        siz_elem=np.array([lab[i][4]],dtype=np.float32)
        siz.append(siz_elem)
        cit_elem=np.array([lab[i][5]],dtype=np.int32)
        cit.append(cit_elem)
        grw5_elem=[]
        for j in range(0,5):
        	#el1=np.array([lab[i][6][j]], dtype=np.float32)
        	el1=lab[i][6][j]
        	#print "el1 ",el1
        	grw5_elem.append([el1])	
        grw5_elem=np.array(grw5_elem, dtype=np.float32)	
        #grw5_elem = np.array([lab[i][6]], dtype=np.float32)
        grw5.append(grw5_elem)

    return grp,feat,tag,tar,grw,siz,cit,grw5


def g_get_sample(length,start):
    feat=[]
    tag=[]
    tar=[]
    grp=[]

    lab = getlist()

    for i in range(start,start+length):
        elem=lab[i][1]
        tag_elem=np.array(elem[:TAG_DIM],dtype=np.float32)
        feat_all=elem[TAG_DIM:]
        feat_elem=[feat_all[j:j + H_DIMS] for j in xrange(0, len(feat_all), H_DIMS)]
        feat_elem=np.array(feat_elem, dtype=np.float32)
        feat.append(feat_elem)
        tag.append(tag_elem)
        tar_elem=np.array([lab[i][2]],dtype=np.float32)
        tar.append(tar_elem)
        grp.append(lab[i][0])
    return grp,feat,tag,tar

