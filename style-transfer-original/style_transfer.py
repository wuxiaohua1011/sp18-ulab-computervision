import time
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import utils
from utils import postp


def load_images(image_dir=None):
    """
    1) Load each image in img_paths. Use Image in the PIL library.
    2) Apply the transformation to each image
    3) Convert each image from size C x W x H to 1 x C x W x H (same as batch size 1)
    4) Wrap each Tensor in a Variable
    5) Return a tuple of (style_image_tensor, content_image_tensor)
    """
    image_dir = image_dir or utils.image_dir
    img_paths = [image_dir + 'vangogh_starry_night.jpg', image_dir + 'Tuebingen_Neckarfront.jpg']
    
    img_starry_night = Image.open(img_paths[0])
    img_neckarfront = Image.open(img_paths[1])
    
    
    
    img_starry_night = utils.prep(img_starry_night)
    img_neckarfront = utils.prep(img_neckarfront)
    
    img_starry_night.unsqueeze_(0)
    img_neckarfront.unsqueeze_(0)
    
    style_image_tensor = Variable(img_starry_night, requires_grad = False)
    content_image_tensor = Variable(img_neckarfront, requires_grad = False)
    
    return style_image_tensor, content_image_tensor
    

def generate_pastiche(content_image):
    """
    Clone the content_image and return wrapped as a Variable with
    requires_grad=True
    """
    return Variable(content_image.data.clone(), requires_grad = True)


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        """ Saves input variables and initializes objects

        Keyword arguments:
        target - the feature matrix of content_image
        weight - the weight applied to this loss (refer to the formula)
        """
        super(ContentLoss, self).__init__()

        self.loss = nn.MSELoss()
        self.target= target
        self.weight = weight
        
    def forward(self, x):
        """ Calculate the content loss. Refer to the notebook for the formula.

        Keyword arguments:
        x -- a selected output layer of feeding the pastiche through the cnn
        """
        output = self.weight * self.loss(x, self.target)
        return output

class GramMatrix(nn.Module):
    def forward(self, x):
        """ Calculates the batchwise Gram Matrices of x. You will want to
        divide the Gram Matrices by W*H in the end. This will help keep
        values normalized and small.

        Keyword arguments:
        x - a B x C x W x H sized tensor, it should be resized to B x C x W*H
               0 1 2 3 --> 0 1 3 2
        """
        flattened = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return torch.bmm(flattened, flattened.permute(0,2,1)) / (x.shape[2] * x.shape[3])


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        """ Saves input variables and initializes objects

        Keyword arguments:
        target - the Gram Matrix of an arbitrary layer of the cnn output for style_image
        weight - the weight applied to this loss (refer to the formula)
        """
        super(StyleLoss, self).__init__()
        self.target = target
        self.weight = weight
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()


    def forward(self, x):
        """Calculates the weighted style loss. Note that we are comparing STYLE,
        so you will need to find the Gram Matrix for x. You will not need to do so
        for target, since it is stated that it is already a Gram Matrix.

        Keyword arguments:
        x - features of an arbitrary cnn layer by feeding the pastiche
        """
        output = self.weight * self.loss(self.gram(x), self.target)
        return output


def construct_style_loss_fns(vgg_model, style_image, style_layers):
    """Constructs and returns a list of StyleLoss instances - one for each given style layer.
    See vgg.py to see how to extract the given layers from the vgg model. After you've calculated
    the targets, make sure to detach the results by calling detach().

    Keyword arguments:
    vgg_model - the pretrained vgg model. See vgg.py for more details
    style_image - the style image
    style_layers - a list of layers of the cnn output we want.

    """
    
    trained = vgg_model(style_image, style_layers)
    n = [64, 128, 256, 512, 512]
    result = []
    for i in range(0,5):
        result.append(StyleLoss(GramMatrix()(trained[i]).detach(), 1000/(n[i]*n[i])))
    return result
        


def construct_content_loss_fns(vgg_model, content_image, content_layers):
    """Constructs and returns a list of ContentLoss instances - one for each given content layer.
    See vgg.py to see how to extract the given layers from the vgg model. After you've calculated
    the targets, make sure to detach the results by calling detach().

    Keyword arguments:
    vgg_model - the pretrained vgg model. See vgg.py for more details
    content_image - the content image
    content_layers - a list of layers of the cnn output we want.

    """
    
    trained = vgg_model(content_image, content_layers)
    n = [512]
    result = []
    for i in range(0,1):
        result.append(ContentLoss(trained[i].detach(), 1))
    return result


def main():
    """The main method for performing style transfer"""
    max_iter, show_iter = 40, 2
    style_layers = ['r11','r21','r31','r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    # Load up all of the style, content, and pastiche Image
    # Construct the loss functions
    vgg_model = utils.load_vgg()
    style_image, content_image = load_images()
    pastiche = generate_pastiche(content_image)

    style_layers = ['r11','r21','r31','r41', 'r51'] 
    content_layers = ['r42']
    loss_layers = style_layers + content_layers

    style_loss_fns = construct_style_loss_fns(vgg_model, style_image, style_layers) 
    content_loss_fns = construct_content_loss_fns(vgg_model, content_image, content_layers)    
    loss_fns = style_loss_fns + content_loss_fns

    max_iter, show_iter = 20, 2
    optimizer = optim.LBFGS([pastiche])
    n_iter = [0]
    print("entering for loop")
    #while n_iter[0] <= max_iter:
    def closure():
        # Implement the optimization step
        optimizer.zero_grad()
        output = vgg_model(pastiche, loss_layers)
        curr_loss = [los(out) for los, out in zip(loss_fns, output)]
        loss = sum(curr_loss)
        loss.backward()
        n_iter[0] += 1
        if n_iter[0] % show_iter == 0:
            print('Iteration: %d, loss: %f' % (n_iter[0], loss.data[0]))
        return loss
    optimizer.step(closure)

    out_img = postp(pastiche.data[0].cpu().squeeze())
    plt.imshow(out_img)
    plt.show()


if __name__ == '__main__':
    main()
