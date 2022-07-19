import os, sys
import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio

class Dataset(data.Dataset):
    def __init__(self,
        data_in=None,
        num_modal=3,
        debug=True
                ):
        super(Dataset, self).__init__()

        self.data_in=data_in
        self.num_modal=num_modal

        if isinstance(data_in, type(None)):
            self.data_dir=None
            self.data_files=None
        else:
            if isinstance(data_in, str) and os.path.isdir(data_in):
                self.data_dir=data_in
                self.data_files=os.listdir(data_in)
                self.data_files.sort()
            elif isinstance(data_in, str) and os.path.isfile(data_in):
                data_dir, data_file=os.path.split(data_in)
                self.data_dir=data_dir
                self.data_files=[data_file]
            else:
                print("Invalid data_in")
                sys.exit(1)
        
        self.debug=debug

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        if self.debug:
            if isinstance(self.data_files, list):
                print(self.data_files[index])
        
        def load_file(mri_dir, mri_file):
            mri_path=os.path.join(mri_dir, mri_file)
            mri=np.squeeze(sio.loadmat(mri_path)['X'])
            return mri_path, mri
        
        data_out=list()
        if isinstance(self.data_files, list):
            _, data=load_file(self.data_dir, self.data_files[index])
            for i in range(self.num_modal):
                data_out.append(torch.from_numpy(data[i]).T)
        
        return data_out

if __name__ == '__main__':
    rootpath="/Users/xli77/Documents/MISA-pytorch/simulation_data"
    ds=Dataset(data_in=os.path.join(rootpath,"sim_siva.mat"), num_modal=3)
    dl=data.DataLoader(dataset=ds, batch_size=1, shuffle=True)
    for i, data_in in enumerate(dl):
        # import pdb; pdb.set_trace()
        print(len(data_in), data_in[0].shape)