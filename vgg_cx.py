import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

def conv2d(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True))


def conv(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


class VGG19_CX(nn.Module):
    """VGG net used for Contextual loss
    """
    def __init__(self):
        super(VGG19_CX, self).__init__()
        self.conv1_1 = nn.Sequential(conv(3, 64), nn.ReLU())
        self.conv1_2 = nn.Sequential(conv(64, 64), nn.ReLU())
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv2_1 = nn.Sequential(conv(64, 128), nn.ReLU())
        self.conv2_2 = nn.Sequential(conv(128, 128), nn.ReLU())
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv3_1 = nn.Sequential(conv(128, 256), nn.ReLU())
        self.conv3_2 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_3 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_4 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv4_1 = nn.Sequential(conv(256, 512), nn.ReLU())
        self.conv4_2 = nn.Sequential(conv(512, 512), nn.ReLU())

    def load_model(self, model_file):
        vgg19_dict = self.state_dict()
        pretrained_dict = torch.load(model_file)
        vgg19_keys = vgg19_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(vgg19_keys, pretrained_keys):
            vgg19_dict[k] = pretrained_dict[pk]
        self.load_state_dict(vgg19_dict)

    def forward(self, input_images):
        feature = {}
        feature['conv1_1'] = self.conv1_1(input_images)
        feature['conv1_2'] = self.conv1_2(feature['conv1_1'])
        feature['pool1'] = self.pool1(feature['conv1_2'])
        feature['conv2_1'] = self.conv2_1(feature['pool1'])
        feature['conv2_2'] = self.conv2_2(feature['conv2_1'])
        feature['pool2'] = self.pool2(feature['conv2_2'])
        feature['conv3_1'] = self.conv3_1(feature['pool2'])
        feature['conv3_2'] = self.conv3_2(feature['conv3_1'])
        feature['conv3_3'] = self.conv3_3(feature['conv3_2'])
        feature['conv3_4'] = self.conv3_4(feature['conv3_3'])
        feature['pool3'] = self.pool3(feature['conv3_4'])
        feature['conv4_1'] = self.conv4_1(feature['pool3'])
        feature['conv4_2'] = self.conv4_2(feature['conv4_1'])

        return feature

class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX

class LoadImg(Dataset):
    def __init__(self,img,transforms):
        super().__init__()
        self.img = Image.fromarray(img)
        self.transforms = transforms

    def __getitem__(self, idx):
        img = self.img
        if self.transforms is not None:
            img = self.transforms(img)
        img = img.repeat(3,1,1)

        return img

    def __len__(self):
        return 1

def get_loader(batch_size,img):
    # SetRange = transforms.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    T = transforms.Compose([transforms.ToTensor(),SetRange])
    dataset = LoadImg(img, T)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    return dataloader
