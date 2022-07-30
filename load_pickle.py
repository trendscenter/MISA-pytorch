import pickle
import torch
import os
from os.path import exists
import fnmatch
import natsort
from re import search
import pandas as pd

filepath = '/data/users2/dkhosravinezhad1/MISA-pytorch/run'
loss_filepath = '/data/users2/dkhosravinezhad1/MISA-pytorch/slurm_log'
array_number = len(os.listdir(filepath)[2:])
slurm_length = int(len(os.listdir(loss_filepath))/2)
slurm_filename = []
for i in os.listdir(loss_filepath):
  if fnmatch.fnmatch(i, 'output*'):
    slurm_filename.append(i)
slurm_filename = natsort.natsorted(slurm_filename)
slurm_final_loss = [-1]
filename = 'res_sim-siva.p'# input("Insert file directory here: ")
loss = []
seventy_fifth_loss = []
filetype = filename[-2:]
lr = []
epochs = []
batch_size = []
epoch = []
h = 0
for i in range(slurm_length):
  slurm_full = os.path.join(loss_filepath,slurm_filename[i])
  output_exists = exists(slurm_full)
  if os.stat(slurm_full).st_size != 0:
    if i < 10:
      with open(slurm_full, 'r') as lost:
        loss_line = lost.readlines()[-2]
        print("array " + (slurm_full[-19:])[-5] + " final loss: " + loss_line[-27:-19])
        loss.append(loss_line[-27:-19])
        lost.seek(0)
        seventy = len(lost.readlines())
        if seventy >= 78:
          with open(slurm_full, 'r') as lost:
            seven_five = lost.readlines()[78]
            print("array " + (slurm_full[-20:])[-5] + " 75th loss: " + seven_five[-27:-19])
            seventy_fifth_loss.append(seven_five[-27:-19])
            lost.seek(0)
        else:
          print(slurm_full + " does not contain a 75th epoch")
          seventy_fifth_loss.append(None)
      with open(slurm_full, 'r') as lost:
        epoch_iterations = lost.readlines()[4:-1]
        epoch_list = []
        for i in epoch_iterations:
            epoch_list.append(i[-27:-19])
        for i, l in enumerate(epoch_list):
          if search('170', l):
            epoch.append(i+1)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: " + str(i+1))
            break
          elif search('169', l):
            epoch.append(i+1)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: " + str(i+1))
            break
        else:
          epoch.append(400)
          print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: 400")
    elif i < 99:
      with open(slurm_full, 'r') as lost:
        loss_line = lost.readlines()[-2]
        print("array " + (slurm_full[-20:])[-6:-4] + " final loss: " + loss_line[-27:-19])
        loss.append(loss_line[-27:-19])
        lost.seek(0)
        seventy = len(lost.readlines())
        if seventy >= 78:
          with open(slurm_full, 'r') as lost:
            seven_five = lost.readlines()[78]
            print("array " + (slurm_full[-20:])[-6:-4] + " 75th loss: " + seven_five[-27:-19])
            seventy_fifth_loss.append(seven_five[-27:-19])
            lost.seek(0)
        else:
          print(slurm_full + " does not contain a 75th epoch")
          seventy_fifth_loss.append(None)
      with open(slurm_full, 'r') as lost:
        epoch_iterations = lost.readlines()[4:-1]
        epoch_list = []
        for i in epoch_iterations:
            epoch_list.append(i[-27:-19])
        for i, l in enumerate(epoch_list):
          if search('170', l):
            epoch.append(i)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: " + str(i))
            break
          elif search('169', l):
            epoch.append(i)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: " + str(i))
            break
        else:
          epoch.append(400)
          print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: 400")
    else:
      with open(slurm_full, 'r') as lost:
        loss_line = lost.readlines()[-2]
        print("array " + (slurm_full[-21:])[-6:-3] + " final loss: " + loss_line[-27:-19])
        loss.append(loss_line[-27:-19])
        lost.seek(0)
        seventy = len(lost.readlines())
        if seventy >= 78:
          with open(slurm_full, 'r') as lost:
            seven_five = lost.readlines()[78]
            print("array " + (slurm_full[-20:])[-6:-3] + " 75th loss: " + seven_five[-27:-19])
            seventy_fifth_loss.append(seven_five[-27:-19])
            lost.seek(0)
        else:
          print(slurm_full + " does not contain a 75th epoch")
          seventy_fifth_loss.append(None)
      with open(slurm_full, 'r') as lost:
        epoch_iterations = lost.readlines()[4:-1]
        epoch_list = []
        for i in epoch_iterations:
            epoch_list.append(i[-27:-19])
        for i, l in enumerate(epoch_list):
          if search('170', l):
            epoch.append(i)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: " + str(i))
            break
          elif search('169', l):
            epoch.append(i)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: " + str(i))
            break
          else:
            epoch.append(400)
            print("array " + (slurm_full[-20:])[-5] + " MATLAB loss epoch: 400")
  else:
    print(slurm_full + " has no contents. Check error file for problem!")

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
print("final loss list: " + str(loss))
print('75th loss list: ' + str(seventy_fifth_loss))
print('MATLAB loss epoch list: ' + str(epoch))
print("learning rate list length: " + str(len(lr))) 
print("epochs list length: " + str(len(epochs))) 
print("batch size list length: " + str(len(batch_size)))
print("final loss list length: " + str(len(loss)))
print("75th loss list length: " + str(len(seventy_fifth_loss)))
print('MATLAB loss epoch list length: ' + str(len(epoch)))

df = pd.DataFrame(list(zip(lr, epochs, batch_size, loss, seventy_fifth_loss, epoch)),
               columns =['learning rate', 'epochs', 'batch size','final loss','75th loss', 'MATLAB loss epoch'])
df.to_csv('/data/users2/dkhosravinezhad1/MISA-pytorch/slurm_csv/SLURM.csv', index = True)