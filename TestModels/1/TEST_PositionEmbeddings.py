from src import position_embeddings as pe
import matplotlib.pyplot as plt
import torch

timesteps = 1000
time = torch.arange(timesteps)
pos_emb = pe.SinusoidalPositionEmbeddings(dim=500)
emb = pos_emb(time=time)

x = emb.T.to('cpu').detach().numpy().copy()
plt.pcolormesh(x, cmap='RdBu')
plt.ylabel('dimension')
plt.xlabel('time step')
plt.colorbar()
plt.show()
