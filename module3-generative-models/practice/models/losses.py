import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
from util.image_pool import ImagePool
from torch.autograd import Variable
import torchvision.transforms as transforms
###############################################################################
# Functions
###############################################################################

class ContentLoss():
	def initialize(self, loss):
		self.criterion = loss
			
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)

	def __call__(self, fakeIm, realIm):
		return self.get_loss(fakeIm, realIm)

class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		model = model.eval()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model

	def initialize(self, loss):
		with torch.no_grad():
			self.criterion = loss
			self.contentFunc = self.contentFunc()
			# https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/2
			# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
			self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			
	def get_loss(self, fakeIm, realIm):
		fakeIm = (fakeIm + 1) / 2.0 #[-1; 1] -> [0; 1]
		realIm = (realIm + 1) / 2.0
		fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
		realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return torch.mean(loss)

	def __call__(self, fakeIm, realIm):
		return self.get_loss(fakeIm, realIm)

class ContextualLoss(nn.Module):
    def __init__(self, h=0.1):
        super(ContextualLoss, self).__init__()
        self.h = h

    def forward(self, inputs, targets):
        '''
        Implementation of Contextual Loss from https://arxiv.org/pdf/1803.02077.pdf paper.
        :param inputs: Phi(X)
        :param targets: Phi(Y)
        :return: contextual loss w.r.t. extracted features from perceptual network Phi.
        '''
        bs, ch = inputs.shape[0:2]
        mu_y = targets.mean(2).mean(2).mean(0).view(1, -1, 1, 1)
        inputs, targets = map(lambda x: x - mu_y, [inputs, targets])
        inputs, targets = map(lambda x: self.l2_normalize(x), [inputs, targets]) #footnote
        inputs, targets = map(lambda x: x.view(bs, ch, -1), [inputs, targets])
        cosine_sim = torch.bmm(inputs.transpose(1, 2), targets)
        d_ij = 1 - cosine_sim
        d_ij_tilda = d_ij / (torch.min(d_ij, dim=2, keepdim=True)[0] + 1e-5) #eq.(2)
        w_ij = torch.exp((1 - d_ij_tilda) / self.h) #eq.(3)
        cx_ij = w_ij / (torch.sum(w_ij, dim=2, keepdim=True) + 1e-8) #eq.(4)
        CX = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1) #contextual similarity, eq.(1)
        CX_loss = -torch.log(CX) #eq.(5)
        return torch.mean(CX_loss)

    def l2_normalize(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


		
class GANLoss(nn.Module):
	def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_l1:
			self.loss = nn.L1Loss()
		else:
			self.loss = nn.BCEWithLogitsLoss()

	def get_target_tensor(self, input, target_is_real):
		"""Create label tensors with the same size as the input.

		Parameters:
		    input (tensor) - - typically the prediction from a discriminator
		    target_is_real (bool) - - if the ground truth label is for real images or fake images

		Returns:
		    A label tensor filled with ground truth label, and with the size of the input
		    """
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor.cuda()

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)


class DiscLoss(nn.Module):
	def name(self):
		return 'DiscLoss'

	def __init__(self):
		super(DiscLoss, self).__init__()

		self.criterionGAN = GANLoss(use_l1=False)
		self.fake_AB_pool = ImagePool(50)
		
	def get_g_loss(self,net, fakeB, realB):
		# First, G(A) should fake the discriminator
		pred_fake = net.forward(fakeB)
		return self.criterionGAN(pred_fake, 1)
		
	def get_loss(self, net, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.pred_fake = net.forward(fakeB.detach())
		self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

		# Real
		self.pred_real = net.forward(realB)
		self.loss_D_real = self.criterionGAN(self.pred_real, 1)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		return self.loss_D

	def __call__(self, net, fakeB, realB):
		return self.get_loss(net, fakeB, realB)


class RelativisticDiscLoss(nn.Module):
	def name(self):
		return 'RelativisticDiscLoss'

	def __init__(self):
		super(RelativisticDiscLoss, self).__init__()

		self.criterionGAN = GANLoss(use_l1=False)
		self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
		self.real_pool = ImagePool(50)

	def get_g_loss(self, net, fakeB, realB):
		# First, G(A) should fake the discriminator
		self.pred_fake = net.forward(fakeB)

		# Real
		self.pred_real = net.forward(realB)
		errG = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 0) +
					   self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
		return errG

	def get_loss(self, net, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.fake_B = fakeB.detach()
		self.real_B = realB
		self.pred_fake = net.forward(fakeB.detach())
		self.fake_pool.add(self.pred_fake)

		# Real
		self.pred_real = net.forward(realB)
		self.real_pool.add(self.pred_real)

		# Combined loss
		self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 1) +
					   self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
		return self.loss_D

	def __call__(self, net, fakeB, realB):
		return self.get_loss(net, fakeB, realB)

class RelativisticDiscLossLS(nn.Module):
	def name(self):
		return 'RelativisticDiscLossLS'

	def __init__(self):
		super(RelativisticDiscLossLS, self).__init__()

		self.criterionGAN = GANLoss(use_l1=True)
		self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
		self.real_pool = ImagePool(50)

	def get_g_loss(self, net, fakeB, realB):
		# First, G(A) should fake the discriminator
		self.pred_fake = net.forward(fakeB)

		# Real
		self.pred_real = net.forward(realB)
		errG = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) +
				torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
		return errG

	def get_loss(self, net, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.fake_B = fakeB.detach()
		self.real_B = realB
		self.pred_fake = net.forward(fakeB.detach())
		self.fake_pool.add(self.pred_fake)

		# Real
		self.pred_real = net.forward(realB)
		self.real_pool.add(self.pred_real)

		# Combined loss
		self.loss_D = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) +
					   torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
		return self.loss_D

	def __call__(self, net, fakeB, realB):
		return self.get_loss(net, fakeB, realB)
		
class DiscLossLS(DiscLoss):
	def name(self):
		return 'DiscLossLS'

	def __init__(self):
		super(DiscLossLS, self).__init__()
		self.criterionGAN = GANLoss(use_l1=True)
		
	def get_g_loss(self,net, fakeB, realB):
		return DiscLoss.get_g_loss(self,net, fakeB)
		
	def get_loss(self, net, fakeB, realB):
		return DiscLoss.get_loss(self, net, fakeB, realB)
		
class DiscLossWGANGP(DiscLossLS):
	def name(self):
		return 'DiscLossWGAN-GP'

	def __init__(self):
		super(DiscLossWGANGP, self).__init__()
		self.LAMBDA = 10
		
	def get_g_loss(self, net, fakeB, realB):
		# First, G(A) should fake the discriminator
		self.D_fake = net.forward(fakeB)
		return -self.D_fake.mean()
		
	def calc_gradient_penalty(self, netD, real_data, fake_data):
		alpha = torch.rand(1, 1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)
		
		disc_interpolates = netD.forward(interpolates)

		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
								  create_graph=True, retain_graph=True, only_inputs=True)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
		return gradient_penalty
		
	def get_loss(self, net, fakeB, realB):
		self.D_fake = net.forward(fakeB.detach())
		self.D_fake = self.D_fake.mean()
		
		# Real
		self.D_real = net.forward(realB)
		self.D_real = self.D_real.mean()
		# Combined loss
		self.loss_D = self.D_fake - self.D_real
		gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
		return self.loss_D + gradient_penalty


def get_loss(model):
    if model['feature_loss'] == 'perceptual':
        feature_loss = PerceptualLoss()
        feature_loss.initialize(nn.MSELoss())
    elif model['feature_loss'] == 'contextual':
        feature_loss = ContextualLoss()
    else:
        raise ValueError("FeatureLoss [%s] not recognized." % model['feature_loss'])

    if model['content_loss'] == 'l2':
        content_loss = ContentLoss()
        content_loss.initialize(nn.MSELoss())
    elif model['content_loss'] == 'l1':
        content_loss = ContentLoss()
        content_loss.initialize(nn.L1Loss())
    else:
        raise ValueError("ContentLoss [%s] not recognized." % model['content_loss'])

    if model['disc_loss'] == 'wgan-gp':
        disc_loss = DiscLossWGANGP()
    elif model['disc_loss'] == 'lsgan':
        disc_loss = DiscLossLS()
    elif model['disc_loss'] == 'gan':
        disc_loss = DiscLoss()
    elif model['disc_loss'] == 'ragan':
        disc_loss = RelativisticDiscLoss()
    elif model['disc_loss'] == 'ragan-ls':
        disc_loss = RelativisticDiscLossLS()
    else:
        raise ValueError("GAN Loss [%s] not recognized." % model['disc_loss'])
    return (content_loss, feature_loss), disc_loss