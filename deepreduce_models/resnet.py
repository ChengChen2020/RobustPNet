

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1,isCulled=False,isTopThinned=False,isBottomThinned=False):
		super(BasicBlock, self).__init__()

		self.isCulled = isCulled
		self.isTopThinned = isTopThinned
		self.isBottomThinned = isBottomThinned

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		if (not self.isCulled) and (not self.isTopThinned):
			self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		if (not self.isCulled) and (not self.isBottomThinned):
			self.relu2 = nn.ReLU()

		self.shortcut = nn.Sequential()
			

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		if (not self.isCulled) and (not self.isTopThinned):
			out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += self.shortcut(x)
		if (not self.isCulled) and (not self.isBottomThinned):
			out = self.relu2(out)
		return out



class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, alpha = 1.0, rho = 1.0, isCulled=[False,False,False,False], isThinned=[False,False]):
		super(ResNet, self).__init__()

		self.alpha = alpha
		self.rho = rho
		self.in_planes = int(64*alpha)
		
		self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,stride=int(1//rho), padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.in_planes)

		self.layer1 = self._make_layer(block, int(64*alpha),  num_blocks[0], isCulled[0],isThinned,stride=1)
		self.layer2 = self._make_layer(block, int(128*alpha), num_blocks[1], isCulled[1],isThinned,stride=2)
		self.layer3 = self._make_layer(block, int(256*alpha), num_blocks[2], isCulled[2],isThinned,stride=2)
		self.layer4 = self._make_layer(block, int(512*alpha), num_blocks[3], isCulled[3],isThinned,stride=2)
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(int(512*alpha)*block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, culled_stages_status,thinning_layer,stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride,culled_stages_status,thinning_layer[0],thinning_layer[1]))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		#out = F.relu(self.bn1(self.conv1(x))) 
		out = self.bn1(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avgpool(out)
		out = torch.flatten(out, 1)
		out = self.fc(out)
		return out

## CIFAR-100 models ###
def DRD_C100_230K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,False])

def DRD_C100_115K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_57K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_49K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,True], isThinned=[False,True])

def DRD_C100_29K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_14K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_C100_12K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,True], isThinned=[False,True])

def DRD_C100_7K(num_classes):
	return ResNet(BasicBlock, [2,1,1,1], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])


## TinyImageNet models ##

def DRD_TINY_918K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,False])

def DRD_TINY_459K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_393K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,False])

def DRD_TINY_229K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_197K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,True])

def DRD_TINY_115K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_98K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 1.0, isCulled=[True,True,False,False], isThinned=[False,True])

def DRD_TINY_57K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 0.5, rho = 0.5, isCulled=[True,False,False,False], isThinned=[False,True])

def DRD_TINY_49K(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes, alpha = 1.0, rho = 0.5, isCulled=[True,True,False,False], isThinned=[False,True])


# Define an input tensor
# input_tensor = torch.rand(1, 3, 64, 64).cuda()  # (batch_size, channels, height, width)
# 
# summ = 0
#
#
# def print_input_shape(module, input):
# 	global summ
# 	summ += input[0].flatten().shape[0]
# 	print(f"Input shape: {input[0].shape}")
#
#
# # create the neural network
# net = DRD_TINY_98K(200).cuda()
# net.eval()
#
# # register the forward hook for ReLU activations
# for module in net.modules():
# 	if isinstance(module, nn.ReLU):
# 		module.register_forward_pre_hook(print_input_shape)
#
# net(input_tensor)
# print(summ)
