import torch
import torch.nn as nn

class dcgan_conv(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(dcgan_conv, self).__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self._layers(input)

class dcgan_upconv(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(dcgan_upconv, self).__init__()
        self._layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self._layers(input)

class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self._layers = nn.ModuleList([
            dcgan_conv(input_dim, hidden_dim),
            dcgan_conv(hidden_dim, hidden_dim*2),
            dcgan_conv(hidden_dim*2, hidden_dim*4),
            dcgan_conv(hidden_dim*4, hidden_dim*8),
        ])
        self._out = nn.Sequential(
            nn.Conv2d(hidden_dim*8, latent_dim, 4, 1, 0),
            # nn.BatchNorm2d(latent_dim),
            nn.Tanh()
        )

    def forward(self, input):
        out = input
        out_lst = []
        for m in self._layers:
            out = m(out)
            out_lst.append(out)
        out = self._out(out)  #[N, 128, 1, 1]
        return out, out_lst

class decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64):
        super().__init__()
        self._up = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim*8, 4, 1, 0),
            nn.BatchNorm2d(hidden_dim*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self._layers = nn.ModuleList([
            dcgan_upconv(hidden_dim*8*2, hidden_dim*4),
            dcgan_upconv(hidden_dim*4*2, hidden_dim*2),
            dcgan_upconv(hidden_dim*2*2, hidden_dim),
        ])

        self._out = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*2, output_dim, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input, skip):
        if len(input.shape) == 2:
            input = input.view((*input.shape, 1, 1))

        out = self._up(input)
        skip = skip[::-1]
        for i in range(len(self._layers)):
            out = self._layers[i](torch.cat((out, skip[i]), dim=1))
        out = self._out(torch.cat((out, skip[-1]), dim=1))

        return out

class lstm(nn.Module):
    """
    decoder LSTM part
    """
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        """

        :param input_dim: input dimension of h_t (default: 128)
        :param output_dim: output_dimension of g_t (default: 128)
        :param hidden_dim: hidden dimension of LSTM (default: 128)
        :param n_layers: number of layers of LSTM (default: 2)
        :param batch_size: batch size
        """
        super(lstm, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        # self._batch_size = batch_size
        self._n_layers = n_layers

        self._emb = nn.Linear(input_dim, hidden_dim)
        self._lstm = nn.ModuleList([nn.LSTMCell(hidden_dim, hidden_dim) for i in range(self._n_layers)])
        self._output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def init_hidden(self, batch_size):
        hidden = []
        device = next(self.parameters()).device
        for i in range(self._n_layers):
            hidden.append((torch.zeros(batch_size, self._hidden_dim).to(device),
                           torch.zeros(batch_size, self._hidden_dim).to(device)))
        return hidden

    def forward(self, input, prev_state=None):
        embedded = self._emb(input.view(-1, self._input_dim))
        h_in = embedded
        if prev_state is None:
            prev_state = self.init_hidden(input.shape[0])
        curr_state = []
        for i in range(self._n_layers):
            new_state = self._lstm[i](h_in, prev_state[i])
            curr_state.append(new_state)
            h_in = new_state[0]
        out = self._output(h_in)

        return out, curr_state

class Gaussianlstm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(Gaussianlstm, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._emb = nn.Linear(input_dim, hidden_dim)
        self._lstm = nn.ModuleList([nn.LSTMCell(hidden_dim, hidden_dim) for i in range(n_layers)])
        self._mu = nn.Linear(hidden_dim, output_dim)
        self._logvar = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        hidden = []
        for i in range(self._n_layers):
            hidden.append((torch.zeros(batch_size, self._hidden_dim).to(device),
                           torch.zeros(batch_size, self._hidden_dim).to(device)))
        return hidden

    def forward(self, input, prev_state):
        if len(input.shape) == 4:
            input = input.squeeze(-1).squeeze(-1)
        embedded = self._emb(input)
        h_in = embedded
        curr_state = []
        if prev_state is None:
            prev_state = self.init_hidden(input.shape[0])
        for i in range(self._n_layers):
            new_state = self._lstm[i](h_in, prev_state[i])
            curr_state.append(new_state)
            h_in = new_state[0]
        mu = self._mu(h_in)
        logvar = self._logvar(h_in)
        return mu, logvar, curr_state

if __name__=='__main__':
    dummy = torch.ones((1, 1, 64, 64))
    enc = encoder(1, 128)
    out, out_lst = enc(dummy)
    print(out.shape)

    dec = decoder(128, 1)
    out2 = dec(out, out_lst)
    print(out2.shape)

