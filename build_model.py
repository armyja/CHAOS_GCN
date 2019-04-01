import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res

        return x


# add 3 GCB
# add 3 (plus + B)
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

        return out

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
