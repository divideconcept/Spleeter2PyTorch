import torch
from torch import nn

#from https://github.com/generalwave/spleeter.pytorch/blob/master/spleeter/models/keras_layer.py
from torch.nn import functional
from math import floor, ceil

class Conv2dKeras(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2dKeras, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            0, dilation, groups,
            bias, padding_mode)
        self.keras_mode = padding

    def _padding_size(self, size, idx):
        output = torch.div(size[idx] + self.stride[idx] - 1, self.stride[idx], rounding_mode='floor')
        padding = (output - 1) * self.stride[idx] + (self.kernel_size[idx] - 1) * self.dilation[idx] + 1 - size[idx]
        padding = max(0, padding)
        return padding

    def forward(self, x):
        if self.keras_mode == 'same':
            size = x.shape[2:]
            row = self._padding_size(size, 0)
            col = self._padding_size(size, 1)
            x = functional.pad(x, [floor(col / 2), ceil(col / 2), floor(row / 2), ceil(row / 2)])

        return super(Conv2dKeras, self).forward(x)


class ConvTranspose2dKeras(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        assert output_padding == 1
        super(ConvTranspose2dKeras, self).__init__(
            in_channels, out_channels, kernel_size, 1,
            dilation * (kernel_size - 1), 0, groups, bias,
            dilation, padding_mode)

        self.keras_kernel_size = kernel_size
        self.keras_stride = stride
        self.keras_padding = padding
        self.keras_output_padding = output_padding
        self.keras_dilation = dilation

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert output_size is None
        output_padding = self.keras_output_padding

        padding = self.keras_dilation * (self.keras_kernel_size - 1) - self.keras_padding

        b, c, h, w = x.shape
        h = h + (h - 1) * (self.keras_stride - 1) + 2 * padding + output_padding
        w = w + (w - 1) * (self.keras_stride - 1) + 2 * padding + output_padding
        hs, he = padding + output_padding, h - padding
        ws, we = padding + output_padding, w - padding
        newx = torch.zeros((b, c, h, w), dtype=x.dtype, device=x.device)
        newx[:, :, hs:he:self.keras_stride, ws:we:self.keras_stride] = x

        return functional.conv_transpose2d(
            newx, self.weight, self.bias, self.stride, self.padding,
            0, self.groups, self.dilation)


#from https://github.com/tuan3w/spleeter-pytorch/
#modified to support keras block (see above) and elu activation
def down_block(in_filters, out_filters, elu=False, keras=True):
    return Conv2dKeras(
        in_filters, out_filters, kernel_size=5, stride=2, padding='same'
        ) if keras else nn.Conv2d(
        in_filters, out_filters, kernel_size=5, stride=2, padding=2), nn.Sequential(
            nn.BatchNorm2d(out_filters, track_running_stats=True, eps=1e-3, momentum=0.01),
            nn.ELU(alpha=1) if elu else nn.LeakyReLU(0.2)
        )


def up_block(in_filters, out_filters, dropout=False, elu=False, keras=True):
    layers = [
        ConvTranspose2dKeras(
        in_filters, out_filters, kernel_size=5, stride=2, padding=2, output_padding=1
        ) if keras else nn.ConvTranspose2d(
        in_filters, out_filters, kernel_size=5, stride=2, padding=2, output_padding=1
        ),
        nn.ELU(alpha=1) if elu else nn.ReLU(),
        nn.BatchNorm2d(out_filters, track_running_stats=True, eps=1e-3, momentum=0.01)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, elu=False, keras=True):
        super(UNet, self).__init__()
        self.down1_conv, self.down1_act = down_block(2, 16, elu)
        self.down2_conv, self.down2_act = down_block(16, 32, elu)
        self.down3_conv, self.down3_act = down_block(32, 64, elu)
        self.down4_conv, self.down4_act = down_block(64, 128, elu)
        self.down5_conv, self.down5_act = down_block(128, 256, elu)
        self.down6_conv, self.down6_act = down_block(256, 512, elu)

        self.up1 = up_block(512, 256, dropout=False, elu=elu)
        self.up2 = up_block(512, 128, dropout=False, elu=elu)
        self.up3 = up_block(256, 64, dropout=False, elu=elu)
        self.up4 = up_block(128, 32, elu=elu)
        self.up5 = up_block(64, 16, elu=elu)
        self.up6 = up_block(32, 1, elu=elu)
        self.up7 = nn.Sequential(Conv2dKeras(1, 2, kernel_size=4, dilation=2, padding='same'
                   ) if keras else nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding='same')) #must also be sequential to properly assign weights

    def forward(self, x):
        d1_conv = self.down1_conv(x)
        d1 = self.down1_act(d1_conv)

        d2_conv = self.down2_conv(d1)
        d2 = self.down2_act(d2_conv)

        d3_conv = self.down3_conv(d2)
        d3 = self.down3_act(d3_conv)

        d4_conv = self.down4_conv(d3)
        d4 = self.down4_act(d4_conv)

        d5_conv = self.down5_conv(d4)
        d5 = self.down5_act(d5_conv)

        d6_conv = self.down6_conv(d5)
        d6 = self.down6_act(d6_conv)

        u1 = self.up1(d6_conv)
        u2 = self.up2(torch.cat([d5_conv, u1], axis=1))
        u3 = self.up3(torch.cat([d4_conv, u2], axis=1))
        u4 = self.up4(torch.cat([d3_conv, u3], axis=1))
        u5 = self.up5(torch.cat([d2_conv, u4], axis=1))
        u6 = self.up6(torch.cat([d1_conv, u5], axis=1))
        u7 = self.up7(u6)
        return u7

