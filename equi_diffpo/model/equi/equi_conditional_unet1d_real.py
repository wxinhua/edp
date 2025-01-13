from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat
from equi_diffpo.model.diffusion.conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.common.rotation_transformer import RotationTransformer


class EquiDiffusionUNetVel(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim, #64
            local_cond_dim=local_cond_dim, #None
            global_cond_dim=global_cond_dim, #256
            diffusion_step_embed_dim=diffusion_step_embed_dim, #128
            down_dims=down_dims, #[512, 1024, 2048]
            kernel_size=kernel_size, #5
            n_groups=n_groups, #8
            cond_predict_scale=cond_predict_scale #True
        )
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.order = self.N
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )


    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            8 * [self.group.trivial_repr]
        )
    
    # matrix
    def getOutput(self, conv_out):
        action = conv_out[:, :8] 
        return action
    
    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        joint_position = act[:, :8]

        return nn.GeometricTensor(joint_position, self.getOutFieldType())
    
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        B, T = sample.shape[:2]
        sample = rearrange(sample, "b t d -> (b t) d")
        sample = self.getActionGeometricTensor(sample)
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        # print(f"enc_a_out shape:{enc_a_out.shape}")
        # print(f"timestep shape:{timestep.shape}")
        # print(f"global_cond shape:{global_cond.shape}")
        
        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)
        #print(f"1 out shape:{out.shape}")
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        #print(f"2 out shape:{out.shape}")
        out = nn.GeometricTensor(out, self.act_type)
        #print(f"3 out shape:{out.shape}")
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        #print(f"4 out shape:{out.shape}")
        out = self.getOutput(out)
        #print(f"5 out shape:{out.shape}")
        out = rearrange(out, "(b t) n -> b t n", b=B)
        #print(f"6 out shape:{out.shape}")
        return out
