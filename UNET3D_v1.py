import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchsummary import summary

################################################################################
################################################################################

class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()
        self.best_loss = 1000000

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        r"""
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file to be restored.")

        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        # Restore model weights
        self.load_state_dict(ckpt_dict['model_state_dict'])

        # Restore optimizer status if existing. Evaluation doesn't need this
        # TODO return optimizer?????
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # Return global step
        return ckpt_dict['epoch']

    def save_checkpoint(self,
                        directory,
                        epoch, loss,
                        optimizer=None,
                        name=None):
        r"""
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            epoch (int): The training. epoch
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':
                self.state_dict(),
            'optimizer_state_dict':
                optimizer.state_dict() if optimizer is not None else None,
            'epoch':
                epoch
        }

        # Save the file with specific name
        if name is None:
            name = "{}_{}_epoch.pth".format(
                os.path.basename(directory),  # netD or netG
                'last')

        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST.pth".format(
                os.path.basename(directory))
            torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()

################################################################################
################################################################################

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base_n_filter, DO):
        super(UNetEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=DO)

        self.conv3d_c1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.norm_lrelu_c1 = self.norm_lrelu(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.norm_lrelu_c2 = self.norm_lrelu(self.base_n_filter*2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.norm_lrelu_c3 = self.norm_lrelu(self.base_n_filter*4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.norm_lrelu_c4 = self.norm_lrelu(self.base_n_filter*8)

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))
    
    def norm_lrelu(self, feat_in):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU())
    
    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    ##########
    def forward(self, x):
        out = self.conv3d_c1(x)
        residual_1 = out
        out = self.lrelu_conv_c1(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.norm_lrelu_c1(out)

        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.norm_lrelu_c2(out)
        context_2 = out

        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.norm_lrelu_c3(out)
        context_3 = out

        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.norm_lrelu_c4(out)
        context_4 = out
        
        skips = [context_1,context_2,context_3,context_4]

        return skips


class UNetBottleneck(nn.Module):
    def __init__(self, base_n_filter, DO):
        super(UNetBottleneck, self).__init__()
        
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=DO)

        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_c5 = self.norm_lrelu(self.base_n_filter * 16)
        
    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))
    
    def norm_lrelu(self, feat_in):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU())

    ##########
    def forward(self, x):
        residual_5 = x
        out = self.norm_lrelu_conv_c5(x)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_c5(out)

        return out


class UNetDecoder(nn.Module):
    def __init__(self,  n_classes, base_n_filter, order):
        super(UNetDecoder, self).__init__()
        
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.order = order

        self.lrelu = nn.LeakyReLU()

        if order ==0:
            self.norm_lrelu_l0 = self.norm_lrelu(self.base_n_filter * 8)
            self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,bias=False)
            self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        elif ((self.order>0) and (self.order<4)):
            # 1 2 3
            pot = 5-order
            self.conv_norm_lrelu_l = self.conv_norm_lrelu(self.base_n_filter * (2**pot), self.base_n_filter * (2**pot))
            self.conv3d_red_l = nn.Conv3d(self.base_n_filter * (2**pot), self.base_n_filter * (2**(pot-1)), kernel_size=1, stride=1, padding=0,bias=False)
            self.norm_lrelu_l = self.norm_lrelu(self.base_n_filter * (2**(pot-1)))
            self.transconv_norm_lrelu_l = self.transconv_norm_lrelu(self.base_n_filter * (2**(pot-1)),self.base_n_filter * (2**(pot-2)))
    
        elif order==4:
            factor_filt=2
            self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * factor_filt, self.base_n_filter * factor_filt)
            self.conv3d_l4 = nn.Conv3d(self.base_n_filter * factor_filt, self.n_classes, kernel_size=1, stride=1, padding=0,bias=False)

    def transconv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.ConvTranspose3d(feat_in, feat_out, kernel_size= 2, stride=2, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())
    
    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())
    
    def norm_lrelu(self, feat_in):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU())

    ##########
    def forward(self, x):
        
        ####################
        if self.order == 0:
            out = self.norm_lrelu_l0(x)
            out = self.conv3d_l0(out)
            out = self.norm_lrelu_l0(out)
            ds=out
        
        elif ((self.order>0) and (self.order<4)):
            out = self.conv_norm_lrelu_l(x)
            ds = out
            out = self.conv3d_red_l(out)
            out = self.norm_lrelu_l(out)
            out = self.transconv_norm_lrelu_l(out)
            
        elif self.order==4:
            out = self.conv_norm_lrelu_l4(x)
            out = self.conv3d_l4(out)
            ds=out
            
        return out, ds

class UNet3D_BranchCT(BaseModel):
    
    def __init__(self, in_channels, n_classes, base_n_filter=8, DO=0.5):
        super(UNet3D_BranchCT, self).__init__()
        
        self.encoder_CT = UNetEncoder(in_channels, base_n_filter, DO)
        self.conv3d_c4_to_bot_CT = nn.Conv3d(base_n_filter * 8, base_n_filter * 16, kernel_size=3, stride=2, padding=1,bias=False)
        self.bottleneck_CT = UNetBottleneck(base_n_filter, DO)
        self.convtrans_bot_to_c4_CT = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size= 2, stride=2, bias=False)
        self.decoder0_CT = UNetDecoder(n_classes, base_n_filter, order=0)
        self.decoder1_CT = UNetDecoder(n_classes, base_n_filter, order=1)
        self.decoder2_CT = UNetDecoder(n_classes, base_n_filter, order=2)
        self.decoder3_CT = UNetDecoder(n_classes, base_n_filter, order=3)
        self.decoder4_CT = UNetDecoder(n_classes, base_n_filter, order=4)
        self.ds2_1x1_conv3d_CT = nn.Conv3d(base_n_filter * 8, n_classes, kernel_size=1, stride=1, padding=0,bias=False)
        self.ds2_1x1_conv3d_up_CT = nn.ConvTranspose3d(n_classes, n_classes, kernel_size= 2, stride=2, bias=False)
        self.ds3_1x1_conv3d_CT = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds1_ds2_sum_upscale_ds3_sum_upscale_CT = nn.ConvTranspose3d(n_classes, n_classes, kernel_size= 2, stride=2, bias=False)
                                        
    def forward(self, x):

        ########################################
        # CT
        ########################################
        [x_1,x_2,x_3,x_4] = self.encoder_CT(x)
        bottle_in_CT = self.conv3d_c4_to_bot_CT(x_4)
        bottle_out_CT = self.bottleneck_CT(bottle_in_CT)
        decoder_in_CT =  self.convtrans_bot_to_c4_CT(bottle_out_CT)
        
        ########################################
        out,_ = self.decoder0_CT(decoder_in_CT)
        out = torch.cat([out, x_4], dim=1)
        out,ds1 = self.decoder1_CT(out)
        out = torch.cat([out, x_3], dim=1)
        out,ds2 = self.decoder2_CT(out)
        out = torch.cat([out, x_2], dim=1)
        out,ds3 = self.decoder3_CT(out)
        out = torch.cat([out, x_1], dim=1)
        out_pred,_ = self.decoder4_CT(out)
        
        ########################################
        ds2_1x1_conv = self.ds2_1x1_conv3d_CT(ds2)
        ds1_ds2_sum_upscale = self.ds2_1x1_conv3d_up_CT(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d_CT(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.ds1_ds2_sum_upscale_ds3_sum_upscale_CT(ds1_ds2_sum_upscale_ds3_sum)

        seg_layer_CT = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        
        return seg_layer_CT,bottle_in_CT
    
    def test(self,device='cpu'):
        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        print("Unet3D test is complete")


class UNet3D_BranchPT(BaseModel):
    
    def __init__(self, in_channels, n_classes, base_n_filter=8, DO=0.5):
        super(UNet3D_BranchPT, self).__init__()
        
        self.encoder_PT = UNetEncoder(in_channels, base_n_filter, DO)
        self.conv3d_c4_to_bot_PT = nn.Conv3d(base_n_filter * 8, base_n_filter * 16, kernel_size=3, stride=2, padding=1,bias=False)
        self.bottleneck_PT = UNetBottleneck(base_n_filter*2, DO) 
        self.convtrans_bot_to_c4_PT = nn.ConvTranspose3d(base_n_filter * 16, base_n_filter * 8, kernel_size= 2, stride=2, bias=False)
        self.decoder0_PT = UNetDecoder(n_classes, base_n_filter, order=0)
        self.decoder1_PT = UNetDecoder(n_classes, base_n_filter, order=1)
        self.decoder2_PT = UNetDecoder(n_classes, base_n_filter, order=2)
        self.decoder3_PT = UNetDecoder(n_classes, base_n_filter, order=3)
        self.decoder4_PT = UNetDecoder(n_classes, base_n_filter, order=4)
        self.ds2_1x1_conv3d_PT = nn.Conv3d(base_n_filter * 8, n_classes, kernel_size=1, stride=1, padding=0,bias=False)
        self.ds2_1x1_conv3d_up_PT = nn.ConvTranspose3d(n_classes, n_classes, kernel_size= 2, stride=2, bias=False)
        self.ds3_1x1_conv3d_PT = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds1_ds2_sum_upscale_ds3_sum_upscale_PT = nn.ConvTranspose3d(n_classes, n_classes, kernel_size= 2, stride=2, bias=False)
        
        self.conv_norm_lrelu_red_bot = self.conv_norm_lrelu(base_n_filter * 32, base_n_filter * 32)
        self.conv3d_red_red_bot = nn.Conv3d(base_n_filter * 32, base_n_filter * 16, kernel_size=1, stride=1, padding=0,bias=False)
            
    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())
                                        
    def forward(self, x, y):
        ########################################
        # PT
        ########################################
        [PT_1,PT_2,PT_3,PT_4] = self.encoder_PT(x)
        bottle_in_PT = self.conv3d_c4_to_bot_PT(PT_4)
        bottle_in = torch.cat([y, bottle_in_PT], dim=1)
        bottle_out_PT = self.bottleneck_PT(bottle_in)
        bottle_out_reduced = self.conv_norm_lrelu_red_bot(bottle_out_PT)
        bottle_out_reduced = self.conv3d_red_red_bot(bottle_out_reduced)
        decoder_in_PT =  self.convtrans_bot_to_c4_PT(bottle_out_reduced)
        
        ########################################
        out,_ = self.decoder0_PT(decoder_in_PT)
        out = torch.cat([out, PT_4], dim=1)
        out,ds1 = self.decoder1_PT(out) 
        out = torch.cat([out, PT_3], dim=1)
        out,ds2 = self.decoder2_PT(out) 
        out = torch.cat([out, PT_2], dim=1)
        out,ds3 = self.decoder3_PT(out)
        out = torch.cat([out, PT_1], dim=1)
        out_pred,_ = self.decoder4_PT(out) 
        
        ########################################
        ds2_1x1_conv = self.ds2_1x1_conv3d_PT(ds2)
        ds1_ds2_sum_upscale = self.ds2_1x1_conv3d_up_PT(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d_PT(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.ds1_ds2_sum_upscale_ds3_sum_upscale_PT(ds1_ds2_sum_upscale_ds3_sum)

        seg_layer_PT = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale

        return seg_layer_PT
    
    def test(self,device='cpu'):
        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        print("Unet3D test is complete")

class UNet3D_Mirror(BaseModel):
    
    def __init__(self, in_channels=2, n_classes=[17,2], base_n_filter=16, DO=0.6):
        super(UNet3D_Mirror, self).__init__()
        
        self.BranchCT = UNet3D_BranchCT(1, n_classes[0], base_n_filter, DO)
        self.BranchPT = UNet3D_BranchPT(1, n_classes[1], base_n_filter, DO)
            
    def forward(self, x):

        CT0 = x[:,0,:,:,:].unsqueeze(1)
        PT0 = x[:,1,:,:,:].unsqueeze(1)
        
        seg_layer_CT,bottle_in_CT = self.BranchCT(CT0)
        seg_layer_PT = self.BranchPT(PT0,bottle_in_CT)
        
        return seg_layer_PT
    
    def test(self,device='cpu'):
        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        print("Unet3D test is complete")

