import os

test_folders = ['runs/' + d for d in os.listdir('runs') if 'folder' in d]

for tf in test_folders:
    for b in os.listdir(tf):
        path = os.path.join(tf, b)
        files = os.listdir(path)
        qsub_file = [f for f in files if '.qsub' in f]
        if len(qsub_file) > 1:
            raise Exception("Found more than 1 .qsub file in directory " + path)
        if len(qsub_file) == 0:
            raise Exception("No .qsub file in directory " + path)
        qsub_file = qsub_file[0]
        os.system('sbatch ' + path + '/' + qsub_file)
