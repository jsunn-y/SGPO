import torch.nn as nn

class AEModel(nn.Module):
    def __init__(self, pretrained_model, input_dim=None, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.pretrained_model = pretrained_model
        #does not require grad
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        self.latent_dim = latent_dim

        # Encoder: shallow neural network operating on outputs from pretrained_model
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder: simple MLP to reconstruct input from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        #pass through pretrained model to get latent representation
        embedding = self.pretrained_model(x)
        #encode the latent representation
        latent = self.encoder(embedding)
        #decode the latent representation
        reconstruction = self.decoder(latent)
        return embedding, latent, reconstruction
