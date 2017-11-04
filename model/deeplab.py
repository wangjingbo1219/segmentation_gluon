import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import cfg


class ASPP(gluon.HybridBlock):
    def __init__(self, channels):
        super(ASPP, self).__init__()
        with self.name_scope():
            self.layer0 = nn.HybridSequential()
            self.layer0.add(nn.Conv2D(channels, 3, padding=1,
                                      dilation=1), nn.BatchNorm())
            self.layer1 = nn.HybridSequential()
            self.layer1.add(nn.Conv2D(channels, 3, padding=4,
                                      dilation=4), nn.BatchNorm())
            self.layer2 = nn.HybridSequential()
            self.layer2.add(nn.Conv2D(channels, 3, padding=8,
                                      dilation=8), nn.BatchNorm())
            self.layer3 = nn.HybridSequential()
            self.layer3.add(nn.Conv2D(channels, 3, padding=14,
                                      dilation=14), nn.BatchNorm())

    def hybrid_forward(self, F, x):
        result = [
            self.layer0(x),
            self.layer1(x),
            self.layer2(x),
            self.layer3(x),
        ]

        return F.concat(*result, dim=1)


class DeepLab(gluon.HybridBlock):
    def __init__(self, pretrained=False):
        super(DeepLab, self).__init__()
        with self.name_scope():
            resnet = gluon.model_zoo.vision.resnet101_v1(
                pretrained=pretrained).features
            self.layer0 = nn.HybridSequential()
            self.layer0.add(*resnet[0:4])
            self.layer1 = resnet[4]
            self.layer2 = resnet[5]
            self.layer3 = resnet[6]
            self.layer4 = resnet[7]

            # for i in range(len(self.layer3)):
            #     self.layer3[i].body[3]._kwargs['stride'] = (1, 1)
            #     self.layer3[i].body[3]._kwargs['dilate'] = (2, 2)
            #     self.layer3[i].body[3]._kwargs['pad'] = (2, 2)
            # self.layer3[0].downsample[0]._kwargs['stride'] = (1, 1)

            for i in range(len(self.layer4)):
                self.layer4[i].body[3]._kwargs['stride'] = (1, 1)
                self.layer4[i].body[3]._kwargs['dilate'] = (2, 2)
                self.layer4[i].body[3]._kwargs['pad'] = (2, 2)
            self.layer4[0].downsample[0]._kwargs['stride'] = (1, 1)

            self.aspp = ASPP(256)

            self.pred = nn.HybridSequential()
            self.pred.add(
                nn.Conv2D(256, kernel_size=1),
                nn.BatchNorm(),
                nn.Conv2D(cfg.n_class, kernel_size=1)
            )

            self.upsample = nn.Conv2DTranspose(
                cfg.n_class, kernel_size=32, strides=16, padding=8)

    def hybrid_forward(self, F, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = self.pred(x)
        x = self.upsample(x)

        return x


def lkm_test():
    net = DeepLab(pretrained=False)
    net.collect_params().initialize(
        init=mx.initializer.Xavier(magnitude=2.0), ctx=cfg.ctx)
    x = nd.random_uniform(shape=(2, 3, 448, 448))
    y = net(x)
    print(y)


if __name__ == '__main__':
    lkm_test()
