import numpy as np
import nibabel as nib
import scipy.io as sio #delete after test
from scipy import linalg

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

# affine matrix:
affine_matrix = nibDiff.affine

# compute adc:
b0s = diff[:,:,:,bval<100]
b0 = b0s.mean(axis=3)
dwis = diff[:,:,:,bval>=100]

bval_dwi = bval[bval>=100]
b0_reshape_guide = list(b0.shape) + [1] #achieving dimensional consistency
b0_4d = b0.reshape(b0_reshape_guide,order='F') #achieving dimensional consistency

tmp2 = []
for i in bval_dwi:
    tmp2.append(np.full_like(mask,i,dtype='float64')) #same size as mask, appended to a list
bval_dwi_vol = np.stack(tmp2,axis=3) #convert that list to np.ndarray
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
ten = np.reshape(ten_vec.conj().T,(sz[0],sz[1],sz[2],6),order='F')

# compute DWIs from tensor
adc_vec_synth = np.matmul(A,ten_vec)
adc_synth = np.reshape(adc_vec_synth.conj().T,sz,order='F')
dwis_synth = np.exp(-adc_synth * bval_dwi_vol)
for i in range(dwis_synth.shape[3]):
    dwis_synth[:,:,:,i] = dwis_synth[:,:,:,i] * b0

tmp3 = []
for i in range(3):
    tmp3.append(b0)
for i in range(dwis_synth.shape[3]):
    tmp3.append(dwis_synth[:,:,:,i])
diff_synth = np.stack(tmp3,axis=3) #convert list to np.ndarray
for i in range(diff_synth.shape[3]):
    diff_synth[:,:,:,i] = diff_synth[:,:,:,i] * mask
#IMPORTANT NOTE: diff_synth in this python script is different than the
#diff_synth in the MATLAB script (i.e., in this script it's masked, in MATLAB
#it is not)

nib.save(nib.Nifti1Image(diff_synth,None), 'diff_synth.nii')
#nib.save(nib.Nifti1Image(diff_synth,affine_matrix), 'diff_synth.nii')

# compute tensor metrics:
FA_vol = np.zeros(szt)
MD_vol = np.zeros(szt)
RD_vol = np.zeros(szt)
V1_vol = np.zeros(list(szt)+[3])
V2_vol = np.zeros(list(szt)+[3])
V3_vol = np.zeros(list(szt)+[3])
L1_vol = np.zeros(szt)
L2_vol = np.zeros(szt)
L3_vol = np.zeros(szt)


for ii in range(szt[0]):
    for jj in range(szt[1]):
        for kk in range(szt[2]):
            if mask_all[ii,jj,kk]:
                tensor_vox = np.squeeze(ten[ii,jj,kk,:])

                tensor_mtx = np.zeros((3,3))
                tensor_mtx[0,0] = tensor_vox[0]
                tensor_mtx[0,1] , tensor_mtx[1,0] = tensor_vox[1] , tensor_vox[1]
                tensor_mtx[0,2] , tensor_mtx[2,0] = tensor_vox[2] , tensor_vox[2]
                tensor_mtx[1,1] = tensor_vox[3]
                tensor_mtx[1,2] , tensor_mtx[2,1] = tensor_vox[4] , tensor_vox[4]
                tensor_mtx[2,2] = tensor_vox[5]

                D,V = linalg.eig(tensor_mtx)
                MD = np.mean(D)
                FA = np.sqrt(np.sum((D-MD)**2)) / np.sqrt(np.sum(D**2)) * (1.5**0.5)

                FA_vol[ii,jj,kk] = FA
                MD_vol[ii,jj,kk] = MD
                RD_vol[ii,jj,kk] = np.mean(D[0:1])
                V1_vol[ii,jj,kk,:] = V[:,2]
                V2_vol[ii,jj,kk,:] = V[:,1]
                V3_vol[ii,jj,kk,:] = V[:,0]
                L1_vol[ii,jj,kk] = D[2]
                L2_vol[ii,jj,kk] = D[1]
                L3_vol[ii,jj,kk] = D[0]

nib.save(nib.Nifti1Image(FA_vol,None), 'FA.nii')
nib.save(nib.Nifti1Image(MD_vol,None), 'MD.nii')
nib.save(nib.Nifti1Image(RD_vol,None), 'RD.nii')
nib.save(nib.Nifti1Image(L1_vol,None), 'L1.nii')
nib.save(nib.Nifti1Image(L2_vol,None), 'L2.nii')
nib.save(nib.Nifti1Image(L3_vol,None), 'L3.nii')
nib.save(nib.Nifti1Image(V1_vol,None), 'V1.nii')
nib.save(nib.Nifti1Image(V2_vol,None), 'V2.nii')
nib.save(nib.Nifti1Image(V3_vol,None), 'V3.nii')

'''
# test:
FA_vol_dict = sio.loadmat('FA_vol.mat')
FA_vol_mat = FA_vol_dict['FA_vol']

counter = 0
ans2 = FA_vol - FA_vol_mat
ans_flat = ans2.flatten()
test = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        test = False
        counter = counter + 1
if test == True:
    print("You're good for FA_vol!")
else:
    print("Oops!")

print('You messed up ',str(counter),' number of times.')
'''
