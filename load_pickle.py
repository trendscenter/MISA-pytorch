import pickle
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
          return super().find_class(module, name)

filename = input("Insert file directory here: ")
objects = []
filetype = filename[-2:]
if filetype == ".p":
  contents = CPU_Unpickler(filename).load()
  with open(filename, 'rb') as handle:
    b = pickle.load(handle)
    print(b)
  '''if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  torch.cuda.device(device)
  with (open(filename, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
    print(objects)'''
elif filetype == "pt":
  print(torch.load(filename,map_location=torch.device('cpu')))
else:
  while filetype != ".p" or "pt":
    print("Error: Not a pickle file")
    filename = input("Insert file directory here: ")
'''with open('run/19/checkpoints/sim-siva/misa_MAT_siva_s0.pt', 'rb') as handle:
    b = pickle.load(handle)'''
  #/data/users2/dkhosravinezhad1/MISA-pytorch/model/MISAK.py