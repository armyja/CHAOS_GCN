import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GCN_8(nn.Module):
    def __init__(self, c, out_c, k=(7, 7)):  # out_Channel=21 in paper
        super(GCN_8, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding=(3, 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, 3))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, 3))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(3, 0))
        self.prelu = nn.PReLU()
        # self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding=((int(k[0] - 1) / 2), 0))
        # self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, int((k[0] - 1) / 2)))
        # self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, int((k[1] - 1) / 2)))
        # self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(int((k[1] - 1) / 2), 0))

    def forward(self, x):

        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r
        x = self.prelu(x)
        return x


class GCN(nn.Module):
    def __init__(self, c, out_c, k=(7, 7)):  # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding=(3, 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, 3))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, 3))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(3, 0))
        # self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding=((int(k[0] - 1) / 2), 0))
        # self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, int((k[0] - 1) / 2)))
        # self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, int((k[1] - 1) / 2)))
        # self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(int((k[1] - 1) / 2), 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = backbone
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 5, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 5, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 5, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 5, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 5, 1, stride=1, bias=False),
                                             BatchNorm(5),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(25, 5, 1, bias=False)
        self.bn1 = BatchNorm(5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BR_super_7(nn.Module):
    def __init__(self, out_c, dilation=False, BatchNorm=nn.BatchNorm2d):
        super(BR_super_7, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=d_rate, dilation=d_rate)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=d_rate, dilation=d_rate)
        
        self.conv_0_0 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.conv_1_0 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.conv_1_1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv_2_0 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(out_c, out_c, kernel_size=5, padding=2)
        # self.conv_2_1 = ASPP(out_c, 16, BatchNorm=BatchNorm)


    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        
        x_0 = self.conv_0_0(x)
        
        x_1 = self.conv_1_0(x)
        x_1 = self.conv_1_1(x_1)
        
        x_2 = self.conv_2_0(x)
        x_2 = self.conv_2_1(x_2)

        x = x + x_res + x_0 + x_1 + x_2

        return x

class BR(nn.Module):
    def __init__(self, out_c, dilation=False):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=d_rate, dilation=d_rate)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=d_rate, dilation=d_rate)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res

        return x


# add 3 GCB
# add 3 (plus + B)
# dilated conv
# problem: label 4 prob map include label 1's prob
# delete layer_4
# super boundary refine
class FCN_GCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        # self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN_8(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN_8(512, self.num_classes)
        self.gcn3 = GCN_8(1024, self.num_classes)

        self.gcn1_1 = GCN_8(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN_8(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN_8(self.num_classes, self.num_classes)

        # self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR_super_7(num_classes, dilation=True)
        self.br2 = BR_super_7(num_classes, dilation=True)
        self.br3 = BR_super_7(num_classes, dilation=False)
        # self.br4 = BR_super_7(num_classes, dilation=False)
        self.br5 = BR_super_7(num_classes, dilation=False)
        self.br6 = BR_super_7(num_classes, dilation=True)
        self.br7 = BR_super_7(num_classes, dilation=True)

        self.br5_1 = BR_super_7(num_classes, dilation=False)
        self.br6_1 = BR_super_7(num_classes, dilation=True)
        self.br7_1 = BR_super_7(num_classes, dilation=True)

        self.br5_2 = BR_super_7(num_classes, dilation=False)
        self.br6_2 = BR_super_7(num_classes, dilation=True)
        self.br7_2 = BR_super_7(num_classes, dilation=True)

        self.br8 = BR_super_7(num_classes, dilation=True)
        self.br9 = BR_super_7(num_classes, dilation=True)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        # fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        # gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        # gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        # gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_1 = gc_fm3
        # gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        # gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        # gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(gc_fm3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 * gc_fm3)
        gc_fm2_2 = self.br6_1(gc_fm2_1)
        # gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2) * gc_fm3
        # gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1) * gc_fm3
        gc_fm2 = F.interpolate(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 * gc_fm2)
        gc_fm1_2 = self.br7_1(gc_fm1_1)
        # gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2) * gc_fm2
        # gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1) * gc_fm2
        gc_fm1 = F.interpolate(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.interpolate(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)
        out = self.br9(gc_fm1)
        n, c, h, w = out.shape

        if debug is True:
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3_1, viz, patient, slice_index, 'gc_fm3_1')
            # self.heatmap(gc_fm3_2, viz, patient, slice_index, 'gc_fm3_2')
            # self.heatmap(gc_fm3_3, viz, patient, slice_index, 'gc_fm3_3')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3')
            self.heatmap(gc_fm2_1, viz, patient, slice_index, 'gc_fm2_1')
            self.heatmap(gc_fm2_2, viz, patient, slice_index, 'gc_fm2_2')
            self.heatmap(gc_fm2_3, viz, patient, slice_index, 'gc_fm2_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2')
            self.heatmap(gc_fm1_1, viz, patient, slice_index, 'gc_fm1_1')
            self.heatmap(gc_fm1_2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1_3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1')
            self.heatmap(out, viz, patient, slice_index, 'out')

        return out

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))



# add 3 GCB
# add 3 (plus + B)
# dilated conv
# problem: label 4 prob map include label 1's prob
# delete layer_4
# super boundary refine
class FCN_GCN_8(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_8, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        # self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        # self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR_super_7(num_classes, dilation=True)
        self.br2 = BR_super_7(num_classes, dilation=True)
        self.br3 = BR_super_7(num_classes, dilation=False)
        # self.br4 = BR_super_7(num_classes, dilation=False)
        self.br5 = BR_super_7(num_classes, dilation=False)
        self.br6 = BR_super_7(num_classes, dilation=True)
        self.br7 = BR_super_7(num_classes, dilation=True)

        self.br5_1 = BR_super_7(num_classes, dilation=False)
        self.br6_1 = BR_super_7(num_classes, dilation=True)
        self.br7_1 = BR_super_7(num_classes, dilation=True)

        self.br5_2 = BR_super_7(num_classes, dilation=False)
        self.br6_2 = BR_super_7(num_classes, dilation=True)
        self.br7_2 = BR_super_7(num_classes, dilation=True)

        self.br8 = BR_super_7(num_classes, dilation=True)
        self.br9 = BR_super_7(num_classes, dilation=True)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        # fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        # gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        # gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        # gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_1 = gc_fm3
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)
        out = self.br9(gc_fm1)
        n, c, h, w = out.shape

        if debug is True:
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3_1, viz, patient, slice_index, 'gc_fm3_1')
            self.heatmap(gc_fm3_2, viz, patient, slice_index, 'gc_fm3_2')
            self.heatmap(gc_fm3_3, viz, patient, slice_index, 'gc_fm3_3')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3')
            self.heatmap(gc_fm2_1, viz, patient, slice_index, 'gc_fm2_1')
            self.heatmap(gc_fm2_2, viz, patient, slice_index, 'gc_fm2_2')
            self.heatmap(gc_fm2_3, viz, patient, slice_index, 'gc_fm2_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2')
            self.heatmap(gc_fm1_1, viz, patient, slice_index, 'gc_fm1_1')
            self.heatmap(gc_fm1_2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1_3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1')
            self.heatmap(out, viz, patient, slice_index, 'out')

        return out

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))



# add 3 GCB
# add 3 (plus + B)
# dilated conv
# problem: label 4 prob map include label 1's prob
# delete layer_4
# super boundary refine
# 124203
class FCN_GCN_7(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_7, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        # self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        # self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR_super_7(num_classes, dilation=True)
        self.br2 = BR_super_7(num_classes, dilation=True)
        self.br3 = BR_super_7(num_classes, dilation=False)
        # self.br4 = BR_super_7(num_classes, dilation=False)
        self.br5 = BR_super_7(num_classes, dilation=False)
        self.br6 = BR_super_7(num_classes, dilation=True)
        self.br7 = BR_super_7(num_classes, dilation=True)

        self.br5_1 = BR_super_7(num_classes, dilation=False)
        self.br6_1 = BR_super_7(num_classes, dilation=True)
        self.br7_1 = BR_super_7(num_classes, dilation=True)

        self.br5_2 = BR_super_7(num_classes, dilation=False)
        self.br6_2 = BR_super_7(num_classes, dilation=True)
        self.br7_2 = BR_super_7(num_classes, dilation=True)

        self.br8 = BR_super_7(num_classes, dilation=True)
        self.br9 = BR_super_7(num_classes, dilation=True)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        # fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        # gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        # gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        # gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_1 = gc_fm3
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)
        n, c, h, w = out.shape

        return out

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))

# add 3 GCB
# add 3 (plus + B)
# dilated conv
# problem: label 4 prob map include label 1's prob
# delete layer_4
class FCN_GCN_6(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_6, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        # self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        # self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes, dilation=True)
        self.br2 = BR(num_classes, dilation=True)
        self.br3 = BR(num_classes, dilation=False)
        # self.br4 = BR(num_classes, dilation=False)
        self.br5 = BR(num_classes, dilation=False)
        self.br6 = BR(num_classes, dilation=True)
        self.br7 = BR(num_classes, dilation=True)

        self.br5_1 = BR(num_classes, dilation=False)
        self.br6_1 = BR(num_classes, dilation=True)
        self.br7_1 = BR(num_classes, dilation=True)

        self.br5_2 = BR(num_classes, dilation=False)
        self.br6_2 = BR(num_classes, dilation=True)
        self.br7_2 = BR(num_classes, dilation=True)

        self.br8 = BR(num_classes, dilation=True)
        self.br9 = BR(num_classes, dilation=True)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        # fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        # gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        # gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        # gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_1 = gc_fm3
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            # self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)
        n, c, h, w = out.shape

        return out

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))


# add 3 GCB
# add 3 (plus + B)
# dilated conv
# problem: label 4 prob map include label 1's prob
# round /= 5
# 083113
class FCN_GCN_5(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_5, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes, dilation=True)
        self.br2 = BR(num_classes, dilation=True)
        self.br3 = BR(num_classes, dilation=False)
        self.br4 = BR(num_classes, dilation=False)
        self.br5 = BR(num_classes, dilation=False)
        self.br6 = BR(num_classes, dilation=True)
        self.br7 = BR(num_classes, dilation=True)

        self.br5_1 = BR(num_classes, dilation=False)
        self.br6_1 = BR(num_classes, dilation=True)
        self.br7_1 = BR(num_classes, dilation=True)

        self.br5_2 = BR(num_classes, dilation=False)
        self.br6_2 = BR(num_classes, dilation=True)
        self.br7_2 = BR(num_classes, dilation=True)

        self.br8 = BR(num_classes, dilation=True)
        self.br9 = BR(num_classes, dilation=True)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)
        n, c, h, w = out.shape
        out[:, 0, :, :] /= c
        out[:, 4, :, :] *= 1.25
        return out

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))



# add 3 GCB
# add 3 (plus + B)
# 134340
# dilated conv
# problem: label 4 prob map include label 1's prob
class FCN_GCN_4(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_4, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes, dilation=True)
        self.br2 = BR(num_classes, dilation=True)
        self.br3 = BR(num_classes, dilation=False)
        self.br4 = BR(num_classes, dilation=False)
        self.br5 = BR(num_classes, dilation=False)
        self.br6 = BR(num_classes, dilation=True)
        self.br7 = BR(num_classes, dilation=True)

        self.br5_1 = BR(num_classes, dilation=False)
        self.br6_1 = BR(num_classes, dilation=True)
        self.br7_1 = BR(num_classes, dilation=True)

        self.br5_2 = BR(num_classes, dilation=False)
        self.br6_2 = BR(num_classes, dilation=True)
        self.br7_2 = BR(num_classes, dilation=True)

        self.br8 = BR(num_classes, dilation=True)
        self.br9 = BR(num_classes, dilation=True)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)
        n, c, h, w = out.shape

        return out

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))

# add 3 GCB
# add 3 (plus + B)
class FCN_GCN_3(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_3, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)

        self.br5_1 = BR(num_classes)
        self.br6_1 = BR(num_classes)
        self.br7_1 = BR(num_classes)

        self.br5_2 = BR(num_classes)
        self.br6_2 = BR(num_classes)
        self.br7_2 = BR(num_classes)

        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm3_1)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm2_1)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm1_1)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))



# add 3 GCB
# add 3 (plus + B)
class FCN_GCN_2(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)

        self.br5_1 = BR(num_classes)
        self.br6_1 = BR(num_classes)
        self.br7_1 = BR(num_classes)

        self.br5_2 = BR(num_classes)
        self.br6_2 = BR(num_classes)
        self.br7_2 = BR(num_classes)

        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3_3 = self.br5_2(gc_fm3_2 + gc_fm4)
        gc_fm3 = F.upsample(gc_fm3_3, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2_3 = self.br6_2(gc_fm2_2 + gc_fm3)
        gc_fm2 = F.upsample(gc_fm2_3, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1_3 = self.br7_2(gc_fm1_2 + gc_fm2)
        gc_fm1 = F.upsample(gc_fm1_3, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))

# add 3 GCB
class FCN_GCN_2(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_1, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)

        self.gcn1_1 = GCN(self.num_classes, self.num_classes)  # gcn_i after layer-1
        self.gcn2_1 = GCN(self.num_classes, self.num_classes)
        self.gcn3_1 = GCN(self.num_classes, self.num_classes)

        self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)

        self.br5_1 = BR(num_classes)
        self.br6_1 = BR(num_classes)
        self.br7_1 = BR(num_classes)

        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)

        gc_fm3_1 = self.br5(gc_fm3 + gc_fm4)
        gc_fm3_2 = self.br5_1(self.gcn3_1(gc_fm3_1))
        gc_fm3 = F.upsample(gc_fm3_2, fm2.size()[2:], mode='bilinear', align_corners=True)

        gc_fm2_1 = self.br6(gc_fm2 + gc_fm3)
        gc_fm2_2 = self.br6_1(self.gcn2_1(gc_fm2_1))
        gc_fm2 = F.upsample(gc_fm2_2, fm1.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1_1 = self.br7(gc_fm1 + gc_fm2)
        gc_fm1_2 = self.br7_1(self.gcn1_1(gc_fm1_1))
        gc_fm1 = F.upsample(gc_fm1_2, pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))


class FCN_GCN_0(nn.Module):
    def __init__(self, num_classes):
        super(FCN_GCN_0, self).__init__()
        self.num_classes = num_classes  # 21 in paper

        resnet = models.resnet50(pretrained=True)
        # input = 256x256
        self.conv1 = resnet.conv1  # 7x7,64, stride=2 o/p = 128x128
        self.bn0 = resnet.bn1  # BatchNorm2d(64)?
        self.relu = resnet.relu
        self.layer1 = resnet.layer1  # res-2 o/p = 64x64,256
        self.layer2 = resnet.layer2  # res-3 o/p = 32x32,512
        self.layer3 = resnet.layer3  # res-4 o/p = 16x16,1024
        self.layer4 = resnet.layer4  # res-5 o/p = 8x8,2048

        self.gcn1 = GCN(256, self.num_classes)  # gcn_i after layer-1
        self.gcn2 = GCN(512, self.num_classes)
        self.gcn3 = GCN(1024, self.num_classes)
        self.gcn4 = GCN(2048, self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_c / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.5),
            nn.Conv2d(in_c / 2, self.num_classes, 1),

        )

    def forward(self, x, debug=False, viz=None, patient=None, slice_index=None):
        # input = x  # 256
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x  # 128
        fm1 = self.layer1(x)  # 64
        fm2 = self.layer2(fm1)  # 32
        fm3 = self.layer3(fm2)  # 16
        fm4 = self.layer4(fm3)  # 8

        gc_fm1 = self.br1(self.gcn1(fm1))  # 64
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        if debug is True:
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_0')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm2_0')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm3_0')
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm4_0')

        gc_fm4 = F.upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.upsample(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        # gc_fm3 = F.upsample(self.br5(gc_fm3), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.upsample(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.upsample(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)  # 128

        gc_fm1 = F.upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        if debug is True:
            self.heatmap(gc_fm4, viz, patient, slice_index, 'gc_fm1_4')
            self.heatmap(gc_fm3, viz, patient, slice_index, 'gc_fm1_3')
            self.heatmap(gc_fm2, viz, patient, slice_index, 'gc_fm1_2')
            self.heatmap(gc_fm1, viz, patient, slice_index, 'gc_fm1_1')

        out = self.br9(gc_fm1)

        return out

    def heatmap(self, input, viz, patient, slice_index, name):
        n, c, h, w = input.shape
        fm1 = input.view(-1, h, w)
        c, h, w = fm1.shape
        for i in range(c):
            viz.heatmap(fm1[i], opts=dict(title=f'{patient + 1}_{slice_index + 1}_{name}_input_class_{i}'))
