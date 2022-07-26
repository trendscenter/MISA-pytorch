import pickle
import torch
import os
from os.path import exists

filepath = '/data/users2/dkhosravinezhad1/MISA-pytorch/run'
array_number = len(os.listdir(filepath)[2:])
filename = 'res_sim-siva.p'# input("Insert file directory here: ")
objects = []
filetype = filename[-2:]
lr = []
epochs = []
batch_size = []
for i in range(array_number):
  full_filename = os.path.join(filepath,str(i), filename)
  file_exists = exists(full_filename)
  if file_exists:
    if i < 10:
      j = full_filename[-16]
      with open(full_filename, 'rb') as handle:
        b = pickle.load(handle)
      lr.append(b['lr']) 
      epochs.append(b['epochs'])
      batch_size.append(b['batch_size'])
      print("array " + str(j) + " learning rate = " + str(lr[int(j)]))
      print("array " + str(j) + " epochs = " + str(epochs[int(j)]))
      print("array " + str(j) + " batch size = " + str(batch_size[int(j)]))
    else:
      j = full_filename[-17:-15] 
      with open(full_filename, 'rb') as handle:
        b = pickle.load(handle)
      lr.append(b['lr']) 
      epochs.append(b['epochs'])
      batch_size.append(b['batch_size'])
      print("array " + str(j) + " learning rate = " + str(lr[i-1]))
      print("array " + str(j) + " epochs = " + str(epochs[i-1]))
      print("array " + str(j) + " batch size = " + str(batch_size[i-1]))
  elif filetype == "pt":
    print(torch.load(full_filename,map_location=torch.device('cpu')))
  else:
    print(full_filename + " does not exist or is corrupted.")
print("learning rate list:" + str(lr)) 
print("epochs list: " + str(epochs)) 
print("batch size list: " + str(batch_size))