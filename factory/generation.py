import torch
import torch.nn as nn
import ipdb
import torch.nn.functional as F

class CVAEModel(nn.Module):
    def __init__(self,encoder, hidden_dim, latent_dim,seq_length, num_layers=2):
        super(CVAEModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder - ESM2
        self.encoder = encoder# ESM2Model.from_pretrained("facebook/esm2", num_hidden_layers=num_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        # self.decoder = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True,bidirectional=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.out = nn.Linear(hidden_dim, self.encoder.config.vocab_size) # Adjust vocab size

        # self.encoder_projector = EncoderCNN(seq_length)
        self.decoder_projector = DecoderCNN(latent_dim,hidden_dim,seq_length)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch): # 不知道合不合理，还需要再学一学
        # Encoding
        encoder_outputs = self.encoder(**batch)
        hidden_state = encoder_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token
        mu = self.fc_mu(hidden_state)
        log_var = self.fc_var(hidden_state)

        # Get latent vector
        z = self.reparameterize(mu, log_var)

        mem = self.decoder_projector(z)
        # Decoding
        ipdb.set_trace()
        # decoder_output, _ = self.decoder(encoder_outputs.last_hidden_state,mem)
        decoder_output = self.decoder(tgt= batch['input_ids'],
                                     memory=encoder_outputs.last_hidden_state)
        
        reconstruction = self.out(decoder_output)

        return reconstruction, mu, log_var


class DecoderCNN(nn.Module):
    def __init__(self,input_dim, hidden_dim, seq_length):
        super(DecoderCNN, self).__init__()
        self.seq_length = seq_length

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 计算上采样的比例
        self.upsample_scale = seq_length

    def forward(self, x):
        # 增加一个维度来适配卷积层的输入要求
        x = x.unsqueeze(-1)

        # 卷积和激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 上采样到指定的序列长度
        x = F.interpolate(x, size=self.seq_length, mode='linear', align_corners=True)

        # 调整维度以匹配输出格式
        x = x.transpose(1, 2)

        return x

class EncoderCNN(nn.Module):
    def __init__(self,seq_length):
        super(EncoderCNN, self).__init__()
        self.seq_length = seq_length

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=seq_length, out_channels=seq_length // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=seq_length // 2, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # 调整维度以匹配卷积层的输入格式
        x = x.transpose(1, 2)

        # 卷积和激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 移除多余的维度
        x = x.squeeze()

        return x

def generate_sequence(cvae_model, seq_length, device='cpu'):
    cvae_model.eval()  # Set the model to evaluation mode

    # Sample from standard normal distribution
    z = torch.randn(1, cvae_model.latent_dim).to(device)

    # Placeholder for generated sequence
    generated_sequence = torch.zeros(1, 1, cvae_model.latent_dim).to(device)

    for _ in range(seq_length):
        # Decode the current sequence
        decoder_output = cvae_model.decoder(generated_sequence, memory=z)
        next_element_logits = cvae_model.out(decoder_output)

        # Select the next element (e.g., using argmax for simplicity)
        next_element = next_element_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append the next element to the sequence
        generated_sequence = torch.cat([generated_sequence, next_element.unsqueeze(-1)], dim=1)

    return generated_sequence

