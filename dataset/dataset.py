import os, sys
import torch
import torch.utils.data as data
import numpy as np
import nibabel as nib

class Dataset(data.Dataset):
    def __init__(self,
        smri_in=None,
        fmri_in=None,
        dmri_in=None,
        debug=True
                ):
        super(Dataset, self).__init__()

        # sMRI
        self.smri_in=smri_in
        if isinstance(smri_in, type(None)):
            self.smri_dir=None
            self.smri_files=None
        else:
            if isinstance(smri_in, str) and os.path.isdir(smri_in):
                self.smri_dir=smri_in
                self.smri_files=os.listdir(smri_in)
                self.smri_files.sort()
            elif isinstance(smri_in, str) and os.path.isfile(smri_in):
                smri_dir, smri_file=os.path.split(smri_in)
                self.smri_dir=smri_dir
                self.smri_files=[smri_file]
            else:
                print("Invalid smri_in")
                sys.exit(1)

        # fMRI
        self.fmri_in=fmri_in
        if isinstance(fmri_in, type(None)):
            self.fmri_dir=None
            self.fmri_files=None
        else:
            if isinstance(fmri_in, str) and os.path.isdir(fmri_in):
                self.fmri_dir=fmri_in
                self.fmri_files=os.listdir(fmri_in)
                self.fmri_files.sort()
            elif isinstance(fmri_in, str) and os.path.isfile(fmri_in):
                fmri_dir, fmri_file=os.path.split(fmri_in)
                self.fmri_dir=fmri_dir
                self.fmri_files=[fmri_file]
            else:
                print("Invalid fmri_in")
                sys.exit(1)

        # dMRI
        self.dmri_in=dmri_in
        if isinstance(dmri_in, type(None)):
            self.dmri_dir=None
            self.dmri_files=None
        else:
            if isinstance(dmri_in, str) and os.path.isdir(dmri_in):
                self.dmri_dir=dmri_in
                self.dmri_files=os.listdir(dmri_in)
                self.dmri_files.sort()
            elif isinstance(dmri_in, str) and os.path.isfile(dmri_in):
                dmri_dir, dmri_file=os.path.split(dmri_in)
                self.dmri_dir=dmri_dir
                self.dmri_files=[dmri_file]
            else:
                print("Invalid dmri_in")
                sys.exit(1)

        self.current_smri_nii=None
        self.current_fmri_nii=None
        self.current_dmri_nii=None
        self.debug=debug

    def __len__(self):
        return len(self.smri_files)

    def __getitem__(self, index):
        if self.debug:
            if isinstance(self.smri_files, list):
                print(self.smri_files[index])
            if isinstance(self.fmri_files, list):
                print(self.fmri_files[index])
            if isinstance(self.dmri_files, list):
                print(self.dmri_files[index])

        def load_nii(mri_dir, mri_file):
            mri_nii=nib.load(os.path.join(mri_dir, mri_file))
            mri=np.array(mri_nii.get_fdata(), dtype=np.float32)
            # 0-1 Normalization
            mri=(mri-mri.min())/(mri.max()-mri.min())
            mri=torch.from_numpy(mri)
            return mri_nii, mri
        
        Out=list()
        if isinstance(self.smri_files, list):
            smri_nii, smri=load_nii(self.smri_dir, self.smri_files[index])
            Out.append(smri)
            self.current_smri_nii=smri_nii

        if isinstance(self.fmri_files, list):
            fmri_nii, fmri=load_nii(self.fmri_dir, self.fmri_files[index])
            Out.append(fmri)
            self.current_fmri_nii=fmri_nii

        if isinstance(self.dmri_files, list):
            dmri_nii, dmri=load_nii(self.dmri_dir, self.dmri_files[index])
            Out.append(dmri)
            self.current_dmri_nii=dmri_nii

        if len(Out)==1:
            Out=Out[0]
        else:
            Out=tuple(Out)
        
        return Out


if __name__ == '__main__':
    rootpath="../data"
    ds=Dataset(smri_in=os.path.join(rootpath,"smri"), fmri_in=os.path.join(rootpath,"fmri"), dmri_in=os.path.join(rootpath,"dmri"))
    dl=data.DataLoader(dataset=ds, batch_size=1, shuffle=True)
    for i, (smri_in, fmri_in, dmri_in) in enumerate(dl):
        print(smri_in.shape, fmri_in.shape, dmri_in.shape)