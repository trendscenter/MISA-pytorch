import pickle


objects = []
with (open("run/19/checkpoints/sim-siva/misa_MAT_siva_s0.pt", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
            '''Exception has occurred: UnpicklingError
A load persistent id instruction was encountered,
but no persistent_load function was specified.
  File "/data/users2/dkhosravinezhad1/MISA-pytorch/pickle.py", line 8, in <module>
    objects.append(pickle.load(openfile))'''
        except EOFError:
            break

with open('run/19/checkpoints/sim-siva/misa_MAT_siva_s0.pt', 'rb') as handle:
    b = pickle.load(handle)

'''Error message: Exception has occurred: UnpicklingError
A load persistent id instruction was encountered,
but no persistent_load function was specified.
  File "/data/users2/dkhosravinezhad1/MISA-pytorch/pickle.py", line 8, in <module>
    objects.append(pickle.load(openfile))'''