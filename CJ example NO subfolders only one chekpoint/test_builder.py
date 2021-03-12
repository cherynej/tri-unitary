import numpy as np
import os
import sys
#import shutil
#print('hi')

template_file='test.template'
template_contents=open(template_file,'r').read()

narg = len(sys.argv) -1
L = sys.argv[1]
eps=sys.argv[2]
W=sys.argv[3]
num_states=sys.argv[4]
tmax=sys.argv[5]
niter = sys.argv[6]
cput = "42:00:00"
mem = '12gb'

#print('bye')

print('%d %.6f %d' %(int(L),float(eps),float(W)))

Run_name_base = 'folder_triunitary_prethermalization_canonical_typicality_L%d_eps%.6f_W%.6f'%(int(L),float(eps),float(W))
Run_name_file = 'file_triunitary_prethermalization_canonical_typicality_L%d_eps%.6f_W%.6f'%(int(L),float(eps),float(W))
Scratch_dir_Base ='/home/groups/vkhemani/cheryne/test/runs/'+str(Run_name_base)
#files_path='/home/groups/vkhemani/cheryne/tri-unitary_prethermalization_canonical_typicality/runs/'
#files_to_copy=['triunitary_prethermalization_canonicaltyp.py', 'SpinLibraryCherynePython3.py']

num = 0
while os.path.exists(Scratch_dir_Base+'/b'+str(num)):
    num = num+1
d = '/b' + str(num) + '/'
Scratch_dir = Scratch_dir_Base + d
Run_name = Run_name_base+d

if not os.path.exists(Scratch_dir):
    os.makedirs(Scratch_dir)

#for f in files_to_copy:
#    shutil.copyfile(files_path + f, Scratch_dir_Base + d + f)

# replace template with qsub
qsub_file=template_file.replace('test.template', Run_name_file + '.qsub')
fout=open(Scratch_dir+'/'+qsub_file,'w')

contents=template_contents.replace('###RN', Run_name)
contents=contents.replace('###L', L)
contents=contents.replace('###eps', eps)
contents=contents.replace('###W', W)
contents=contents.replace('###num_states', num_states)
contents=contents.replace('###tmax', tmax)
contents=contents.replace('###niter', niter)

contents=contents.replace('###CPUT', cput)
contents=contents.replace('###mem', mem)
contents=contents.replace('###run_number', 'b' + str(num))

fout.write(contents)
fout.close()
