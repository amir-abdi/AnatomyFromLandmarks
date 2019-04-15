import torch


class _G(torch.nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()

        # This architecture only supports cube_len of 70, 140 and 141 voxel cubes
        assert args.cube_len == 140 or args.cube_len == 70 or args.cube_len == 141

        self.args = args
        self.cube_len = args.cube_len
        self.landmark_size = self.args.num_landmarks * 3
        net_size = args.net_size

        if args.num_fc > 0:
            fc_layers = []
            prev_layer_size = self.landmark_size

            # if num_fc, output tensor size of fc layers are: init * 2, init * 8, init * 64
            for i in range(args.num_fc):
                fc_layers.append(torch.nn.Linear(prev_layer_size, prev_layer_size * (2 ** (i + 1))))
                prev_layer_size *= (2 ** (i + 1))
            self.layer0 = torch.nn.Sequential(*fc_layers)

            self.layer1 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.landmark_size, int(self.cube_len * net_size), kernel_size=1,
                                         stride=1,
                                         bias=args.bias,
                                         padding=(0, 0, 0)),
                torch.nn.CELU())
        else:
            self.layer1 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.landmark_size, int(self.cube_len * net_size), kernel_size=4,
                                         stride=2,
                                         bias=args.bias,
                                         padding=(0, 0, 0)),
                torch.nn.CELU())

        self.inception_layerE = InceptionE(int(self.cube_len * net_size))

        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(int(self.cube_len * net_size), int(self.cube_len * net_size), kernel_size=4,
                                     stride=2,
                                     bias=args.bias,
                                     padding=(1, 1, 1)),
            torch.nn.CELU())

        self.inception_layerB = InceptionB(int(self.cube_len * net_size))

        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(int(self.cube_len * net_size * 2), int(self.cube_len), kernel_size=4,
                                     stride=2,
                                     bias=args.bias,
                                     padding=(1, 1, 1)),
            torch.nn.CELU())

        if args.batch_norm:
            self.layer5 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=4,
                                         stride=2,
                                         bias=args.bias,
                                         padding=(2, 2, 2)),
                torch.nn.BatchNorm3d(self.cube_len),
                torch.nn.CELU())
        else:
            self.layer5 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.cube_len, self.cube_len, kernel_size=4,
                                         stride=2,
                                         bias=args.bias,
                                         padding=(2, 2, 2)),
                torch.nn.CELU())

        if self.cube_len == 70:
            last_pad = (2, 2, 2)
        elif self.cube_len == 140 or self.cube_len == 141:
            last_pad = (1, 1, 1)

        self.out_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=args.bias, padding=last_pad),
            torch.nn.Sigmoid())

    def forward(self, x):
        if self.args.num_fc > 0:
            out = x.view(-1, self.landmark_size)
            out = self.layer0(out)
            out = out.view(-1, self.landmark_size, 4, 4, 4)
        else:
            out = x.view(-1, self.landmark_size, 1, 1, 1)

        out = self.layer1(out)
        out = self.inception_layerE(out)
        out = self.layer2(out)
        out = self.inception_layerB(out)
        out = self.layer4(out)

        # The current architecture only supports cube_len of 70 and 140
        if self.args.cube_len == 140:
            out = self.layer5(out)

        out = self.out_layer(out)
        return out


class BasicConvTranspose3d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvTranspose3d, self).__init__()
        self.conv = torch.nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.functional.F.celu(x, inplace=True)


class InceptionB(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConvTranspose3d(in_channels, 184, kernel_size=4, stride=2)

        self.branch3x3dbl_1 = BasicConvTranspose3d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConvTranspose3d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConvTranspose3d(96, 96, kernel_size=4, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = torch.functional.F.interpolate(x, size=(18, 18, 18))

        outputs = [branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)


class InceptionE(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConvTranspose3d(in_channels, 32, kernel_size=1)

        self.branch3x3_1 = BasicConvTranspose3d(in_channels, 38, kernel_size=1)
        self.branch3x3_2a = BasicConvTranspose3d(38, 38, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch3x3_2b = BasicConvTranspose3d(38, 38, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.branch3x3_2c = BasicConvTranspose3d(38, 38, kernel_size=(1, 1, 3), padding=(0, 0, 1))

        self.branch3x3dbl_1 = BasicConvTranspose3d(in_channels, 44, kernel_size=1)
        self.branch3x3dbl_2 = BasicConvTranspose3d(44, 38, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConvTranspose3d(38, 38, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch3x3dbl_3b = BasicConvTranspose3d(38, 38, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.branch3x3dbl_3c = BasicConvTranspose3d(38, 38, kernel_size=(1, 1, 3), padding=(0, 0, 1))

        self.branch_pool = BasicConvTranspose3d(in_channels, 20, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
            self.branch3x3_2c(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
            self.branch3x3dbl_3c(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = torch.functional.F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionA(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConvTranspose3d(in_channels, 60, kernel_size=1)

        self.branch5x5_1 = BasicConvTranspose3d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConvTranspose3d(48, 60, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConvTranspose3d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConvTranspose3d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConvTranspose3d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConvTranspose3d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = torch.functional.F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
