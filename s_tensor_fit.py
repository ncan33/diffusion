import numpy as np
import nibabel as nib
import scipy.io as sio #delete after test
from scipy import linalg
import sys #delete after test

# load data
fpDiff = '../diff18/mwu100307_diff.nii.gz'
fpMask = '../diff18/mwu100307_diff_mask.nii.gz'
fpBval = '../diff18/mwu100307_diff.bval'
fpBvec = '../diff18/mwu100307_diff.bvec'

# first, NIfTI images:
nibDiff = nib.load(fpDiff)
nibMask = nib.load(fpMask)
diff = np.array(nibDiff.dataobj) #<class numpy.ndarray>
mask = np.array(nibMask.dataobj) #<class numpy.ndarray>

# second, b-values and diffusion tensors:
bval = np.loadtxt(fpBval) #<class numpy.ndarray>
bvec = np.loadtxt(fpBvec) #<class numpy.ndarray>

# compute adc:
b0s = diff[:,:,:,bval<100]
b0 = b0s.mean(axis=3)
dwis = diff[:,:,:,bval>=100]

bval_dwi = bval[bval>=100]
b0_reshape_guide = list(b0.shape) + [1] #achieving dimensional consistency
b0_4d = b0.reshape(b0_reshape_guide,order='F') #achieving dimensional consistency

if np.max(bval_dwi) == np.min(bval_dwi): #for when the b-values are the same for each DWI, i.e. all 1000
    adc = -np.log(np.divide(dwis, b0_4d)) / bval_dwi[0]
else: #for when the b-values are different for each DWI, we must do more than just scalar division
    stack_this = []
    for i in bval_dwi:
        stack_this.append(np.full_like(mask,i,dtype='float64')) #same size as mask, appended to a list
    bval_dwi_vol = np.stack(stack_this,axis=3) #convert that list to np.ndarray
    adc = np.divide(-np.log(np.divide(dwis, b0_4d)), bval_dwi_vol)

# clean up adc
mask1 = np.sum(np.isnan(adc),axis=3) == False
mask2 = np.sum(np.isinf(adc),axis=3) == False
mask_all = (mask * mask1 * mask2) > 0.1 #i do not think "> 0.1" is necessary

not_mask_all = np.logical_not(mask_all)
mask_map = np.repeat(not_mask_all[:,:,:,np.newaxis],adc.shape[3],axis=3)
adc[mask_map] = 0; #setting NaN and inf values in adc to 0

# compute tensor
bvec_dwi = bvec[bval >= 100,:]
A = np.array([ \
    bvec_dwi[:,0] * bvec_dwi[:,0], \
    2 * bvec_dwi[:,0] * bvec_dwi[:,1], \
    2 * bvec_dwi[:,0] * bvec_dwi[:,2], \
    bvec_dwi[:,1] * bvec_dwi[:,1], \
    2 * bvec_dwi[:,1] * bvec_dwi[:,2], \
    bvec_dwi[:,2] * bvec_dwi[:,2] \
    ]).T #transposes this ndarray

sz = adc.shape
szt = sz[:3]
adc_vec = np.reshape(adc,(np.prod(szt),sz[3]),order='F')
ten_vec_tuple = np.linalg.lstsq(A,adc_vec.conj().T,rcond=None)
ten_vec = ten_vec_tuple[0]

'''
with open(fpBval) as f, open(fpBvec) as g:
    strBval = f.read()
    strBvec = g.read()

nibabel notes:
print(nibDiff.shape)
print(nibDiff.get_data_dtype() == np.dtype(np.float64))
'''

'''
adc_dict = sio.loadmat('adc.mat')
adc_vec_dict = sio.loadmat('adc_vec.mat')
ten_vec_dict = sio.loadmat('ten_vec.mat')
adc_mat = adc_dict['adc']
adc_vec_mat = adc_vec_dict['adc_vec']
ten_vec_mat = ten_vec_dict['ten_vec']


ans = adc - adc_mat
ans_flat = ans.flatten()
we_gud = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        we_gud = False
if we_gud == True:
    print("You're good for adc buddy!")
else:
    print("Oof")

ans = adc_vec - adc_vec_mat
ans_flat = ans.flatten()
we_gud = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        we_gud = False
        print('fuck')
if we_gud == True:
    print("You're good for adc_vec buddy!")
else:
    print("Oof")

counter = 0
ans = ten_vec_mat - ten_vec
ans_flat = ans.flatten()
we_gud = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        we_gud = False
        counter = counter + 1
if we_gud == True:
    print("You're good for ten_vec buddy!")
else:
    print("Oof")

print('You fucked up ',str(counter),' number of times.')
'''


'''
stack_this = []
for i in bval_dwi:
    stack_this.append(np.full_like(mask,i,dtype='float64')) #same size as mask, appended to a list
bval_dwi_vol = np.stack(stack_this,axis=3) #convert that list to np.ndarray
adc = np.divide(-np.log(np.divide(dwis, b0_4d)), bval_dwi_vol)
'''
