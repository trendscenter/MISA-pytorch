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
h = 0
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
    elif i > 99:
      j = int(full_filename[-18:-15]) 
      with open(full_filename, 'rb') as handle:
        b = pickle.load(handle)
      lr.append(b['lr']) 
      epochs.append(b['epochs'])
      batch_size.append(b['batch_size'])
      print("array " + str(j) + " learning rate = " + str(lr[j-h]))
      print("array " + str(j) + " epochs = " + str(epochs[j-h]))
      print("array " + str(j) + " batch size = " + str(batch_size[j-h]))
    else:
      j = int(full_filename[-17:-15]) 
      with open(full_filename, 'rb') as handle:
        b = pickle.load(handle)
      lr.append(b['lr']) 
      epochs.append(b['epochs'])
      batch_size.append(b['batch_size'])
      print("array " + str(j) + " learning rate = " + str(lr[j-h]))
      print("array " + str(j) + " epochs = " + str(epochs[j-h]))
      print("array " + str(j) + " batch size = " + str(batch_size[j-h]))
  elif filetype == "pt":
    print(torch.load(full_filename,map_location=torch.device('cpu')))
  else:
    print(full_filename + " does not exist or is corrupted.")
    h += 1
print("learning rate list:" + str(lr)) 
print("epochs list: " + str(epochs)) 
print("batch size list: " + str(batch_size))
print("learning rate list length: " + str(len(lr))) 
print("epochs list length: " + str(len(epochs))) 
print("batch size list length: " + str(len(batch_size)))