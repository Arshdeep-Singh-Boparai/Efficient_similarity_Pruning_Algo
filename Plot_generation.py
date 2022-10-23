#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 23:30:46 2022

@author: arshdeep
"""

import matplotlib.pyplot as plt
import numpy as np

#%% VGGish...m vs error



C6 = [47.67, 27.72, 14.093, 12.24, 8.672, 8.37, 4.34,3.43,0,0,0,0,0,0]
C5 = [60.94,50.65,25.88,13.53,13.33,11.55,6.33,3.809,0,0,0,0,0,0]
C4 = [110.72,37.38,7.22,7.08,6.92,2.56,2.15,0.85,0,0,0,0,0,0]
C3 = [58.96,55.03,18.80,18.70,17.70,8.24,6.22,4.96,0,0,0,0,0,0]
C2 = [35.30,35.30,18.35,14.06,10.39,9.11,4.39,4.29,0,0,0,0,0,0]
C1 = [15.11,14.05,11.35,11.22,11.10,9.67,9.67,4.90,0,0,0,0,0,0]






m= [1,2,3,4,5,6,7,8,9,10,20,30,40, 50]



layers= ['C1','C2', 'C3','C4','C5','C6']
t = np.arange(0,7)
# plt.figure(figsize=(8,3))
plt.subplots(figsize=(12,6))



plt.plot(C1, color='r',marker='*',linestyle='solid',markersize=15,label = 'C1 ($n$ = 64)')
plt.plot(C2, color='g',marker='o',linestyle='dashed',markersize=15, label ='C2 ($n$ = 128)')
plt.plot(C3, color='blue',marker='p',linestyle='dotted',markersize=15, label = 'C3 ($n$ = 256)')
plt.plot(C4, color='grey',marker='D',linestyle='dashdot',markersize=15, label = 'C4 ($n$ = 256)')
plt.plot(C5, color='orange',marker='X',linestyle=(0, (3, 1, 1, 1)),markersize=15, label = 'C5 ($n$ = 512)')
plt.plot(C6, color='green',marker='v',linestyle=(0, (5, 10)),markersize=15, label = 'C6 ($n$ = 512)')



# plt.tick_params(axis='y', labelcolor=color)
plt.yticks([0,20,40,60,80,100],fontsize=22)
plt.xticks(np.arange(0,15),m,fontsize=22)
plt.ylabel('$\delta$',fontsize =30)
plt.xlabel('$m$', fontsize = 30)
plt.legend(ncol=2,fontsize=25)
# ax2.set_ylim(50,70)
# plt.yticks(fontsize=18)
# plt.ylim(0,51)
# ax2.axis(linestyle='--')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%% DCASE 21.....
m= [1,2,3,4,5,6,7,8,9,10,12,15,21,25,27,30,32]
C1 = [3.446397307644907, 3.2999888188631883, 3.0883678731332336, 3.050700376145296, 2.2888054902762986, 1.8043998386554523, 1.5124565986781493, 1.5061764285527788, 0.9470204045106723, 0.6934056501529987, 0.44035790071762193, 0.287262127875962, 0.28692583203186783, 0.2867560687412479, 0.20915, 0.0]
C2 = [6.311289819990661, 4.316953225148776, 2.3224368305155068, 1.8265118733766355, 1.0249018818230795, 0.8760563603887866, 0.7875614415974669, 0.74570709192808, 0.7251577240721591, 0.6922740966932601, 0.691886244129187, 0.6611928480102649, 0.2950822642825679, 0.2898779686525836, 0.04073, 0.0]
C3 = [7.91670885696328, 5.348923457418032, 4.785998858906464, 4.546756938409254, 4.510914533186975, 4.4715020125913885, 2.9806476159174977, 2.2276515492979336, 1.5862265135001328, 1.3299814631908002, 1.200906072498052, 1.0963616816077397, 1.0651145151602808, 1.031539057962666, 0.6628889035493573, 0.620153957988161, 0.6193782055865458, 0.5592027207993954, 0.5085073443264922, 0.5080881839761755, 0.39934344750115547, 0.344501607319131, 0.3180327206778855, 0.31271552323376817, 0.30392449923099707, 0.3023720417343743, 0.29757633674203415, 0.2110884007178237, 0.1984213349738852, 0.1972879438927059, 0.03881, 0.0]

plt.subplots(figsize=(12,6))
plt.plot(C1, color='r',marker='*',linestyle='solid',markersize=15,label = 'C1 ($n$ = 16)')
plt.plot(C2, color='g',marker='o',linestyle='dashed',markersize=15, label ='C2 ($n$ = 16)')
plt.plot(C3, color='blue',marker='p',linestyle='dotted',markersize=15, label = 'C3 ($n$ = 32)')


lst = np.arange(1,32,2)# + '32'
# plt.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize=22)
plt.xticks(np.arange(0,33,2),lst,fontsize=18)
plt.ylabel('$\delta$',fontsize =30)
plt.xlabel('$m$', fontsize = 30)
plt.legend(ncol=1,fontsize=25)
# ax2.set_ylim(50,70)
# plt.yticks(fontsize=18)
# plt.ylim(0,51)
# ax2.axis(linestyle='--')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


#%% Time versus layers


VGG_time_NYS = [0.00471,0.13,0.40,1.23,2.55,7.40]
VGG_time_CS = [0.15,0.80,2.70,3.51,11.65,16]
layers = ['C1','C2','C3','C4','C5','C6']

 


plt.figure(figsize=(14,6.5))
plt.plot(VGG_time_NYS, color='g',marker='*',linestyle='solid',markersize=15,label = 'With distance matrix approximation (proposed)')
plt.plot(VGG_time_CS, color='orange',marker='o',linestyle='dashed',markersize=15, label ='Without distance matrix approximation')


lst = np.arange(1,32,2)# + '32'
# plt.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize=25)
plt.xticks(np.arange(0,6),layers,fontsize=25)
plt.ylabel('Time (seconds)',fontsize =25)
plt.xlabel('Convolution layer', fontsize = 25)
plt.legend(ncol=1,fontsize=25,framealpha=0)
# ax2.set_ylim(50,70)
# plt.yticks(fontsize=18)
# plt.ylim(0,51)
# ax2.axis(linestyle='--')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%% DCASE TIME VS LAYERS
layers = ['C1','C2','C3']

DCASE21_time_NYS = [0.0018,0.0028,0.0061]
DCASE21_time_CS = [0.0088,0.0104,0.0434]



plt.figure(figsize=(14,6.5))
plt.plot(DCASE21_time_NYS, color='g',marker='*',linestyle='solid',markersize=15,label = 'With distance matrix approximation (proposed)')
plt.plot(DCASE21_time_CS, color='orange',marker='o',linestyle='dashed',markersize=15, label ='Without distance matrix approximation')


lst = np.arange(1,32,2)# + '32'
# plt.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize=25)
plt.xticks(np.arange(0,3),layers,fontsize=25)
plt.ylabel('Time (seconds)',fontsize =25)
plt.xlabel('Convolution layer', fontsize = 25)
plt.legend(ncol=1,fontsize=25,framealpha=0)
# ax2.set_ylim(50,70)
# plt.yticks(fontsize=)
# plt.ylim(0,51)
# ax2.axis(linestyle='--')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%% k vs epsilon, k <= m, DCASE 21


C1 = [3.532415729546738, 2.1890177506321935, 2.0648804081498287, 1.9288150469069627, 1.207601294297645, 1.1642216827262934, 0.7948619630480607, 0.4446075005705799, 0.43129033348393614, 0.2934386071149749, 0.2911783301894342, 0.287262127875962]
C2 = [4.43092916321391, 2.679041567400647, 2.4573968814759812, 1.2473132646336287, 0.9790364731637864, 0.8760563603887866]
C3 = [6.491500067241934, 6.262774151297133, 3.6701997270246896, 3.3712174333294165, 1.8649312270139449, 1.859496470239554, 1.0230765156014074, 0.9120172795784499, 0.7073731604705074, 0.7012150787124182, 0.6770582643979093, 0.6697751829169508, 0.6673105837708866, 0.5252248333156144, 0.5014861687061977, 0.4691924391275909, 0.46915046212219275, 0.4647399929414462, 0.4441812653327352, 0.40080990920349974, 0.39934344750115547]



plt.subplots(figsize=(12,6))
plt.plot(C1, color='r',marker='*',linestyle='solid',markersize=15,label = 'C1 ($m$ =12)')
plt.plot(C2, color='g',marker='o',linestyle='dashed',markersize=15, label ='C2 ($m$ = 6)')
plt.plot(C3, color='blue',marker='p',linestyle='dotted',markersize=15, label = 'C3 ($m$ = 21)')


lst = np.arange(1,22,2)# + '32'
# plt.tick_params(axis='y', labelcolor=color)
plt.yticks(fontsize=25)
plt.xticks(np.arange(0,22,2),lst,fontsize=25)
plt.ylabel('$\delta$',fontsize =30)
plt.xlabel('$k$', fontsize = 30)
plt.legend(ncol=1,fontsize=25)
# ax2.set_ylim(50,70)
# plt.yticks(fontsize=18)
# plt.ylim(0,51)
# ax2.axis(linestyle='--')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%%


C1 = [11.237513035208243, 10.468573348492116, 10.40505244526043, 10.404071858416255, 9.048688989275817, 9.017597428299338, 8.318205497714114, 6.306364247776219, 0.0]
C2 =  [20.90997977878497, 13.463576055299434, 13.4521762032046, 12.711439246737278, 10.252532645141889, 6.358129720562375, 3.233548997895027, 2.769442348173978, 1.414213562366659e-05]
C3 =  [22.34891909733885, 20.528002663039974, 20.44026838727181, 18.660612883134487, 6.885186707077393, 6.155296141900085, 5.4237913431143125, 5.316743707279323, 2.6615545005637854e-05]
C4 = [13.047612872325342, 8.305362893025263, 7.42182974829115, 6.678347991798443, 6.386124562572141, 5.357224078838063, 1.4201791623755922, 0.8786870561585757, 2.111990736253759e-05]
C5 = [55.05186122645962, 27.889990418898513, 11.26070740650645, 9.657651012891115, 9.474929613562013, 8.386899600546018, 3.943226539477397, 3.5101810069116173, 2.7552778449197098e-05]
C6 = [43.92285287057675, 43.003909156606376, 10.45720604932004, 9.575246290557454, 7.163837538252599, 7.157883321165632, 5.059627207181622, 4.623759716546174, 2.787916393478957e-05]





layers= ['C1','C2', 'C3','C4','C5','C6']
t = np.arange(0,7)
# plt.figure(figsize=(8,3))
plt.subplots(figsize=(12,6))



plt.plot(C1, color='r',marker='*',linestyle='solid',markersize=15,label = 'C1 ($m$ = 9)')
plt.plot(C2, color='g',marker='o',linestyle='dashed',markersize=15, label ='C2 ($m$ = 9)')
plt.plot(C3, color='blue',marker='p',linestyle='dotted',markersize=15, label = 'C3 ($m$ = 9)')
plt.plot(C4, color='grey',marker='D',linestyle='dashdot',markersize=15, label = 'C4 ($m$ = 9)')
plt.plot(C5, color='orange',marker='X',linestyle=(0, (3, 1, 1, 1)),markersize=15, label = 'C5 ($m$ = 9)')
plt.plot(C6, color='green',marker='v',linestyle=(0, (5, 10)),markersize=15, label = 'C6 ($m$ = 9)')



# plt.tick_params(axis='y', labelcolor=color)
plt.yticks([0,20,40,60,80,100],fontsize=25)
plt.xticks(np.arange(0,9),np.arange(1,10),fontsize=25)
plt.ylabel('$\delta$',fontsize =30)
plt.xlabel('$k$', fontsize = 30)
plt.legend(ncol=2,fontsize=25)
# ax2.set_ylim(50,70)
# plt.yticks(fontsize=18)
# plt.ylim(0,51)
# ax2.axis(linestyle='--')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


