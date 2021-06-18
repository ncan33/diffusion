#tested with adc[35,35,35,2] = adc[35,35,35,2] + 2*0.0000000001
#adc test wrong -- adc test actually works! trust test and debug

#adc_dict = sio.loadmat('adc.mat')
#adc_vec_dict = sio.loadmat('adc_vec.mat')
#ten_dict = sio.loadmat('ten.mat')
#adc_synth_dict = sio.loadmat('adc_synth.mat')

#adc_mat = adc_dict['adc']
#adc_vec_mat = adc_vec_dict['adc_vec']
#ten_mat = ten_dict['ten']
#adc_synth_mat = adc_synth_dict['adc_synth']

'''
ans = adc - adc_mat
ans_flat = ans.flatten()
test = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        test = False
if test == True:
    print("You're good for adc!")
else:
    print("Oops!")

ans = adc_vec - adc_vec_mat
ans_flat = ans.flatten()
test = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        test = False
        print('mess')
if test == True:
    print("You're good for adc_vec!")
else:
    print("Oops!")


counter = 0
ans = adc_synth_mat - adc_synth
ans_flat = ans.flatten()
test = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        test = False
        counter = counter + 1
if test == True:
    print("You're good for adc_synth!")
else:
    print("Oops!")

print('You messed up ',str(counter),' number of times.')
'''

'''
ans3 = adc - adc_mat
ans_flat = ans3.flatten()
test = True
for i in ans_flat:
    if i>0.0000000001 or i<-0.0000000001:
        test = False
if test == True:
    print("You're good for adc!")
else:
    print("Oops!")
'''

'''
stack_this_as_well = []
for i in range(times_this.shape[3]):
    #times_this[:,:,:,i] = times_this[:,:,:,i] * b0
    stack_this_as_well.append(times_this[:,:,:,i] * b0)
dwis_synth = np.stack(stack_this_as_well,axis=3) #convert list to np.ndarray

stack_this = []
for i in bval_dwi:
    stack_this.append(np.full_like(mask,i,dtype='float64')) #same size as mask, appended to a list
bval_dwi_vol = np.stack(stack_this,axis=3) #convert that list to np.ndarray
adc = np.divide(-np.log(np.divide(dwis, b0_4d)), bval_dwi_vol)
'''

'''
with open(fpBval) as f, open(fpBvec) as g:
    strBval = f.read()
    strBvec = g.read()

nibabel notes:
print(nibDiff.shape)
print(nibDiff.get_data_dtype() == np.dtype(np.float64))

a = np.arange(8).reshape(2,2,2)
b = np.arange(24).reshape(2,2,2,3)
c = np.multiply(a,b)
'''

'''
if np.max(bval_dwi) == np.min(bval_dwi): #for when the b-values are the same for each DWI, i.e. all 1000
    adc = -np.log(np.divide(dwis, b0_4d)) / bval_dwi[0]
else: #for when the b-values are different for each DWI, we must do more than just scalar division
    stack_this = []
    for i in bval_dwi:
        stack_this.append(np.full_like(mask,i,dtype='float64')) #same size as mask, appended to a list
    bval_dwi_vol = np.stack(stack_this,axis=3) #convert that list to np.ndarray
    adc = np.divide(-np.log(np.divide(dwis, b0_4d)), bval_dwi_vol)
'''
