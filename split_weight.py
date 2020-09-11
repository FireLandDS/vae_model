from vae_cnn import VAE
import torch


model = VAE(input_dim=3).cuda()
model.load_state_dict(torch.load('final.pth'))
torch.save(model.encoder.state_dict(), 'encoder.pth')
