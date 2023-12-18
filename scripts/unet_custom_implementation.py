import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # First block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.relu6 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Third block
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.relu9 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fourth block
        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(512, 512, kernel_size=2, stride=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, kernel_size=2, stride=1)
        self.relu12 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fifth and middle block
        self.conv13 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(1024, 1024, kernel_size=2, stride=1)
        self.relu14 = nn.ReLU()
        self.conv15 = nn.Conv2d(1024, 1024, kernel_size=2, stride=1)
        self.relu15 = nn.ReLU()
        # Upsample first
        self.transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv16 = nn.Conv2d(1024, 512, kernel_size=3, stride=1)
        self.relu16 = nn.ReLU()
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.relu17 = nn.ReLU()
        # Second upsample
        self.transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv18 = nn.Conv2d(512, 256, kernel_size=3, stride=1)
        self.relu18 = nn.ReLU()
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.relu19 = nn.ReLU()
        # Third upsample
        self.transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv20 = nn.Conv2d(256, 128, kernel_size=3, stride=1)
        self.relu20 = nn.ReLU()
        self.conv21 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.relu21 = nn.ReLU()
        # Fourth upsample
        self.transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv22 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.relu22 = nn.ReLU()
        self.conv23 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu23 = nn.ReLU()
        # Final output layer
        self.conv24 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()


    def crop_tensor(self, x, y):
      diffY = x.size()[2] - y.size()[2]
      diffX = x.size()[3] - y.size()[3]
      if diffX % 2 != 0:
        left_index = int(diffX / 2)
        right_index = int(diffX / 2) + 1
      else:
        left_index = int(diffX / 2)
        right_index = int(diffX / 2)
      # each_side = torch.ceil(torch.Tensor([diffY / 2])).int()[0]
      return x[:, :, left_index:-right_index, left_index:-right_index]


    def forward(self, x):
        # First block
        c1 = self.conv1(x)
        r1 = self.relu1(c1)
        c2 = self.conv2(r1)
        r2 = self.relu2(c2)
        c3 = self.conv3(r2)
        r3= self.relu3(c3)
        m1 = self.maxpool1(r3)
        # Second block
        c4 = self.conv4(m1)
        r4 = self.relu4(c4)
        c5 = self.conv5(r4)
        r5 = self.relu5(c5)
        c6 = self.conv6(r5)
        r6 = self.relu6(c6)
        m2 = self.maxpool2(r6)
        # Third block
        c7 = self.conv7(m2)
        r7 = self.relu7(c7)
        c8 = self.conv8(r7)
        r8 = self.relu8(c8)
        c9 = self.conv9(r8)
        r9 = self.relu9(c9)
        m3 = self.maxpool3(r9)
        # Fourth block
        c10 = self.conv10(m3)
        r10 = self.relu10(c10)
        c11 = self.conv11(r10)
        r11 = self.relu11(c11)
        c12 = self.conv12(r11)
        r12 = self.relu12(c12)
        m4 = self.maxpool4(r12)
        # Fifth and middle block
        c13 = self.conv13(m4)
        r13 = self.relu13(c13)
        c14 = self.conv14(r13)
        r14 = self.relu14(c14)
        c15 = self.conv15(r14)
        r15 = self.relu15(c15)

        t1 = self.transpose1(r15)
        cropped_1 = self.crop_tensor(r12, t1)
        c_1 = torch.cat([cropped_1, t1], 1)
        c16 = self.conv16(c_1)
        r16 = self.relu16(c16)
        c17 = self.conv17(r16)
        r17 = self.relu17(c17)

        t2 = self.transpose2(r17)
        cropped_2 = self.crop_tensor(r9, t2)
        c_2 = torch.cat([cropped_2, t2], 1)
        c18 = self.conv18(c_2)
        r18 = self.relu18(c18)
        c19 = self.conv19(r18)
        r19 = self.relu19(c19)

        t3 = self.transpose3(r19)
        cropped_3 = self.crop_tensor(r6, t3)
        c_3 = torch.cat([cropped_3, t3], 1)
        c20 = self.conv20(c_3)
        r20 = self.relu20(c20)
        c21 = self.conv21(r20)
        r21 = self.relu21(c21)

        t4 = self.transpose4(r21)
        cropped_4 = self.crop_tensor(r3, t4)
        c_4 = torch.cat([cropped_4, t4], 1)
        c22 = self.conv22(c_4)
        r22 = self.relu22(c22)
        c23 = self.conv23(r22)
        r23 = self.relu23(c23)
        c24 = self.conv24(r23)
        outputs = self.sigmoid(c24)
        return outputs