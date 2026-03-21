from typing import *
import torch
import torch.nn as nn
from torch import Tensor
from .base.linear_group import LinearGroup
from .base.non_linear import *
from .base.norm import *
from mamba_ssm import Mamba as Mamba

class SpatialNetLayer(nn.Module):

    def __init__(
        self,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "LN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full: nn.Module = None,
        attention: str = "mamba(16,4)",
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList(
            [
                new_norm(
                    norms[3],
                    dim_hidden,
                    seq_last=True,
                    group_size=None,
                    num_groups=f_conv_groups,
                ),
                nn.Conv1d(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
                # nn.Tanh()
            ]
        )
        # full-band linear module
        self.norm_full = new_norm(
            norms[5],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=f_conv_groups,
        )
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1),
            nn.SiLU(),
            # nn.Tanh()
        )
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = (
            LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze)
            if full == None
            else full
        )
        self.unsqueeze = nn.Sequential(
            nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1),
            nn.SiLU(),
            # nn.Tanh()
        )
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList(
            [
                new_norm(
                    norms[4],
                    dim_hidden,
                    seq_last=True,
                    group_size=None,
                    num_groups=f_conv_groups,
                ),
                nn.Conv1d(
                    in_channels=dim_hidden,
                    out_channels=dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
                # nn.Tanh()
            ]
        )

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(
            norms[0],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )

        attn_params = attention[6:-1].split(",")
        d_state, mamba_conv_kernel = int(attn_params[0]), int(attn_params[1])
        self.mhsa_f = Mamba(
            d_model=dim_hidden,
            d_state=d_state,
            d_conv=mamba_conv_kernel,
            expand=2,
        )
        # self.mhsa_f = Mamba(
        #     hidden_size=dim_hidden,
        #     state_size=d_state,
        #     conv_kernel=mamba_conv_kernel,
        #     intermediate_size=2*dim_hidden,
        #     time_step_rank=dim_hidden//16
        # )
        # layer_idx=0)

        self.attention = attention
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module

        self.norm_tconvffn = new_norm(
            norms[1],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.tconvffn_b = Mamba(
            d_model=dim_hidden,
            d_state=d_state,
            d_conv=mamba_conv_kernel,
            expand=2,
        )
        # self.tconvffn_b = Mamba(
        #     hidden_size=dim_hidden,
        #     state_size=d_state,
        #     conv_kernel=mamba_conv_kernel,
        #     intermediate_size=2*dim_hidden,
        #     time_step_rank=dim_hidden//16
        # )
        # layer_idx=0)

        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        # x = x + (self._mamba(x, self.mhsa_f, self.norm_mhsa, self.dropout_mhsa)+ self._mamba(x.flip(-2), self.tconvffn_b, self.norm_tconvffn, self.dropout_tconvffn).flip(-2))/2
        x = x + self._mamba(x, self.mhsa_f, self.norm_mhsa, self.dropout_mhsa)
        x = x + self._mamba(
            x.flip(-2), self.tconvffn_b, self.norm_tconvffn, self.dropout_tconvffn
        ).flip(-2)

        return x
    

    def _mamba(self, x: Tensor, mamba, norm: nn.Module, dropout: nn.Module):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)

        x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        # x = nn.functional.tanh(x)
        return dropout(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * T, H, F).contiguous()
        for m in ml:
            if isinstance(m, GroupBatchNorm):  # 用isinstance更安全
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F).permute(0, 3, 1, 2).contiguous()
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.squeeze(self.norm_full(x).permute(0, 2, 3, 1).reshape(B * T, H, F).contiguous())  # [B*T,H',F]
        if self.dropout_full:
            x = x.view(B, T, -1, F)
            x = self.dropout_full(x.transpose(1, 3)).transpose(1, 3)
            x = x.reshape(B * T, -1, F).contiguous()

        x = self.unsqueeze(self.full(x)).view(B, T, H, F).permute(0, 3, 1, 2).contiguous()
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class SpatialNetLayer_nb(nn.Module):

    def __init__(
        self,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "LN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full: nn.Module = None,
        attention: str = "mamba(16,4)",
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]

        attn_params = attention[6:-1].split(",")
        d_state, mamba_conv_kernel = int(attn_params[0]), int(attn_params[1])
        self.full_share = False if full == None else True

        self.norm_mamba_t_f = new_norm(
            norms[0],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.mamba_t_f = Mamba(
            d_model=dim_hidden,
            d_state=d_state,
            d_conv=mamba_conv_kernel,
            expand=2,
        )
        # self.mamba_t_f = Mamba(
        #     hidden_size=dim_hidden,
        #     state_size=d_state,
        #     conv_kernel=mamba_conv_kernel,
        #     intermediate_size=2*dim_hidden,
        # )
        self.dropout_mamba_t_f = nn.Dropout(dropout[0])

        self.norm_mamba_t_b = new_norm(
            norms[1],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.mamba_t_b = Mamba(
            d_model=dim_hidden,
            d_state=d_state,
            d_conv=mamba_conv_kernel,
            expand=2,
        )
        # self.mamba_t_b = Mamba(
        #     hidden_size=dim_hidden,
        #     state_size=d_state,
        #     conv_kernel=mamba_conv_kernel,
        #     intermediate_size=2*dim_hidden,
        # )
        self.dropout_mamba_t_b = nn.Dropout(dropout[1])

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """

        # x = x+(self._mamba(x, self.mamba_t_f, self.norm_mamba_t_f, self.dropout_mamba_t_f)+ self._mamba(x.flip(-2), self.mamba_t_b, self.norm_mamba_t_b, self.dropout_mamba_t_b).flip(-2))/2
        x = x + self._mamba(
            x, self.mamba_t_f, self.norm_mamba_t_f, self.dropout_mamba_t_f
        )
        x = x + self._mamba(
            x.flip(-2), self.mamba_t_b, self.norm_mamba_t_b, self.dropout_mamba_t_b
        ).flip(-2)
        return x

    def _mamba(self, x: Tensor, mamba, norm: nn.Module, dropout: nn.Module):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)

        x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class FiLMConditioner(nn.Module):
    """Produces gamma, beta for Feature-wise Linear Modulation: out = gamma * x + beta."""

    def __init__(self, num_room_params: int, dim_hidden: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden = num_room_params * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(num_room_params, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_hidden * 2),
        )
        self.dim_hidden = dim_hidden

    def forward(self, room_params: Tensor) -> Tuple[Tensor, Tensor]:
        """room_params: [B, num_params] -> gamma, beta: each [B, dim_hidden]"""
        out = self.mlp(room_params)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class Embedding1DConv(nn.Module):
    """1D conv over frequency → ReLU → BN → Conv1D(stride 2) → ReLU → Global Max Pool."""

    def __init__(
        self,
        dim_hidden: int,
        kernel_size: int = 5,
        film_post_conv1: Optional["FiLMConditioner"] = None,
        film_post_conv2: Optional["FiLMConditioner"] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            dim_hidden, dim_hidden, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.conv2 = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=3, stride=2, padding=1)
        self.film_post_conv1 = film_post_conv1
        self.film_post_conv2 = film_post_conv2

    def forward(self, x_CTF: Tensor, room_params: Optional[Tensor] = None) -> Tensor:
        """x_CTF: [B, F, 1, H] -> out: [B, H]."""
        x = x_CTF.squeeze(2).permute(0, 2, 1)  # [B, H, F]
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        if room_params is not None and self.film_post_conv1 is not None:
            gamma, beta = self.film_post_conv1(room_params)
            x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        x = self.conv2(x)
        x = torch.relu(x)
        if room_params is not None and self.film_post_conv2 is not None:
            gamma, beta = self.film_post_conv2(room_params)
            x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        x = x.max(dim=2)[0]  # [B, H] global max pool
        return x


class FuseLayer(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.beta = torch.nn.Parameter(torch.tensor(0.5))
    def forward(self,x,y):
        """
        Args:
            x: shape [B, F, T, H]
            y: shape [B, F, T, H]
        Shape:
            out: shape [B, F, T, H]
        """
        return self.alpha * x + self.beta * y

class BiSpatialNet(nn.Module):

    def __init__(
        self,
        dim_input: int,  # the input dim for each time-frequency point
        dim_output_spch: int,  # the output dim for each time-frequency point
        dim_output_CTF: int,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        num_layers_spch: int,
        num_layers_noise:int,
        num_layers_CTF: int,
        num_angles: int = 181,
        max_rad_value: float = 6.0,
        rad_resolution: float = 0.1,
        encoder_kernel_size: int = 1,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full_share: int = 0,  # share from layer 0
        attention: str = "mhsa(251)",  # mhsa(frames), ret(factor)
        use_film: bool = False,
        num_room_params: int = 8,
        embedding_type: str = "mean",  # "mean" | "variance" | "1dconv"
        embedding_1dconv_kernel: int = 5,
    ):
        super().__init__()
        self.embedding_type = embedding_type

        self.padding_size = (0, (encoder_kernel_size - 1) // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_input,
                out_channels=dim_hidden,
                padding=0,
                kernel_size=(1, encoder_kernel_size),
            ),
            nn.PReLU(),
        )

        full = None
        spch_layers = []
        for l in range(num_layers_spch):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            spch_layers.append(layer)
        self.spch_layers = nn.ModuleList(spch_layers)
        
        full = None
        noise_layers = []
        for l in range(num_layers_noise):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            noise_layers.append(layer)
        self.noise_layers = nn.ModuleList(noise_layers)
        
        full = None
        ctf_layers = []
        for l in range(num_layers_CTF):
            layer = SpatialNetLayer_nb(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            ctf_layers.append(layer)
        self.ctf_layers = nn.ModuleList(ctf_layers)

        self.decoder_spch = nn.Sequential(
            nn.Linear(in_features=dim_hidden, out_features=dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_output_spch),
        )
        self.decoder_rev = nn.Sequential(
            nn.Linear(in_features=dim_hidden, out_features=dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_output_spch),
        )
        self.decoder_CTF = nn.Sequential(
            nn.Linear(in_features=dim_hidden, out_features=dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_output_CTF),
        )
        self.compress_CTF=FuseLayer()

        self.weight_layer = nn.Sequential(
            nn.Linear(in_features=dim_hidden, out_features=dim_hidden),

            nn.LeakyReLU(),
            nn.Linear(in_features=dim_hidden, out_features=1),
            nn.Softmax(dim=2),
        )

        dim_embedding = (
            2 * dim_hidden
            if self.embedding_type == "variance"
            else dim_hidden
        )

        if self.embedding_type == "1dconv":
            if use_film:
                self.film_embedding_conv1 = FiLMConditioner(num_room_params, dim_hidden)
                self.film_embedding_conv2 = FiLMConditioner(num_room_params, dim_hidden)
            else:
                self.film_embedding_conv1 = None
                self.film_embedding_conv2 = None
            self.embedding_1dconv = Embedding1DConv(
                dim_hidden,
                kernel_size=embedding_1dconv_kernel,
                film_post_conv1=self.film_embedding_conv1,
                film_post_conv2=self.film_embedding_conv2,
            )
        else:
            self.embedding_1dconv = None

        num_rad_classes = int(max_rad_value / rad_resolution) + 1
        self.angle_fc1 = nn.Linear(in_features=dim_embedding, out_features=512)
        self.angle_act1 = nn.ReLU()
        self.angle_drop1 = nn.Dropout(p=0.2)
        self.angle_fc2 = nn.Linear(in_features=512, out_features=256)
        self.angle_act2 = nn.ReLU()
        self.angle_drop2 = nn.Dropout(p=0.2)
        self.angle_fc3 = nn.Linear(in_features=256, out_features=num_angles)
        self.radius_head = nn.Linear(in_features=dim_embedding, out_features=num_rad_classes)

        self.use_film = use_film
        if use_film:
            self.film_conditioner = FiLMConditioner(num_room_params, dim_embedding)
            self.film_angle_1 = FiLMConditioner(num_room_params, 512)
            self.film_angle_2 = FiLMConditioner(num_room_params, 256)
            self.film_radius = FiLMConditioner(num_room_params, dim_embedding)
        else:
            self.film_conditioner = None
            self.film_angle_1 = None
            self.film_angle_2 = None
            self.film_radius = None

    def forward(
        self, input: Tensor, return_embedding: bool = False, room_params: Optional[Tensor] = None
    ) -> Tensor:

        input_pad = torch.nn.functional.pad(
            input,
            (
                self.padding_size[1],
                self.padding_size[1],
                self.padding_size[0],
                self.padding_size[0],
            ),
            mode="constant",
            value=0,
        )

        x = self.encoder(input_pad)
        x = x.permute(0, 2, 3, 1)

        B, F, T, H = x.shape
        x_noisy=x
        for i, m in enumerate(self.noise_layers):
            x = m(x)
        x_rev = x
        y_rev = self.decoder_rev(x).permute(0, 3, 1, 2)
            
        for i, m in enumerate(self.spch_layers):
            x = m(x)

        y_spch = self.decoder_spch(x).permute(0, 3, 1, 2)
        
        x=self.compress_CTF(x,x_rev)
        for i, m in enumerate(self.ctf_layers):
            x = m(x)

        # x: [B, F, T, H] -> weight_layer(x): [B, F, T, 1] -> (x * weight).sum(-2): [B, F, H] -> unsqueeze(2): [B, F, 1, H]
        x_CTF = (x * self.weight_layer(x)).sum(-2).unsqueeze(2)  # [B, F, 1, H] = [B, F, 1, dim_hidden]
        
        if return_embedding:
            return x_CTF

        # x_CTF: [B, F, 1, H] -> ctf_embedding for localization heads
        if self.embedding_type == "1dconv":
            ctf_embedding = self.embedding_1dconv(x_CTF, room_params=room_params)  # [B, H]
        elif self.embedding_type == "variance":
            ctf_mean = x_CTF.mean(dim=1).squeeze(1)  # [B, 1, H]
            ctf_std = x_CTF.std(dim=1).squeeze(1)  # [B, 1, H]
            ctf_embedding = torch.cat([ctf_mean, ctf_std], dim=-1).squeeze(1)  # [B, 2*H]
        else:
            ctf_embedding = x_CTF.mean(dim=1).squeeze(1)  # [B, 1, H] -> squeeze to [B, H]
        if self.use_film and self.film_conditioner is not None and room_params is not None:
            gamma, beta = self.film_conditioner(room_params)
            ctf_embedding = gamma * ctf_embedding + beta
        angle_x = self.angle_fc1(ctf_embedding)
        angle_x = self.angle_act1(angle_x)
        if self.use_film and self.film_angle_1 is not None and room_params is not None:
            gamma, beta = self.film_angle_1(room_params)
            angle_x = gamma * angle_x + beta
        angle_x = self.angle_drop1(angle_x)
        angle_x = self.angle_fc2(angle_x)
        angle_x = self.angle_act2(angle_x)
        if self.use_film and self.film_angle_2 is not None and room_params is not None:
            gamma, beta = self.film_angle_2(room_params)
            angle_x = gamma * angle_x + beta
        angle_x = self.angle_drop2(angle_x)
        angle_logits = self.angle_fc3(angle_x)

        radius_in = ctf_embedding
        if self.use_film and self.film_radius is not None and room_params is not None:
            gamma, beta = self.film_radius(room_params)
            radius_in = gamma * radius_in + beta
        radius_logits = self.radius_head(radius_in)
        
        y_CTF = self.decoder_CTF(x_CTF).reshape([B, F, 2, -1]).permute(0, 2, 1, 3)
        
        return y_spch,y_CTF,y_rev,angle_logits,radius_logits

if __name__ == "__main__":
    model = BiSpatialNet(
        dim_input = 2,
        dim_output_spch = 2,
        dim_output_CTF = 120,
        dim_hidden = 96,
        dim_squeeze = 8,
        num_freqs = 257,
        num_layers_spch = 6,
        num_layers_noise = 2,
        num_layers_CTF = 4,
        encoder_kernel_size = 5,
        dropout = [0, 0, 0],
        kernel_size = [5, 3],
        conv_groups = [8, 8],
        norms = ["LN", "LN", "LN", "LN", "LN", "LN"],
        full_share = 0,
        attention = "mamba(16,4)",
    ).cuda()
    


    x = torch.randn((1, 2, 257, 250)).cuda()
    
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(model, display=False) as fcm:
        res = model(x)
        flops_forward_eval = fcm.get_total_flops()
    for k, v in fcm.get_flop_counts().items():
        ss = f"{k}: {{"
        for kk, vv in v.items():
            ss += f" {str(kk)}:{vv}"
        ss += " }"
        print(ss)
    params_eval = sum(param.numel() for param in model.parameters())
    print(
        f"flops_forward={flops_forward_eval/4e9:.2f}G/s, params={params_eval/1e6:.2f} M"
    )
    
