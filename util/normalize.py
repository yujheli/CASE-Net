import torch
from torchvision import transforms
from torch.autograd import Variable
import config

class NormalizeImage(object):
    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange
        self.normalize = transforms.Normalize(mean=config.MEAN,
                                              std=config.STDDEV)
        
    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0                
            sample[key] = self.normalize(sample[key])
        return  sample
    
def normalize_image(image, forward=True, mean=config.MEAN, std=config.STDDEV):
        im_size = image.size()
        mean=torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        std=torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
        if image.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        if isinstance(image, torch.autograd.variable.Variable):
            mean = Variable(mean, requires_grad=False)
            std = Variable(std, requires_grad=False)
        if forward:
            if len(im_size) == 3:
                result = image.sub(mean.expand(im_size)).div(std.expand(im_size))
            elif len(im_size) == 4:
                result = image.sub(mean.unsqueeze(0).expand(im_size)).div(std.unsqueeze(0).expand(im_size))
        else:
            if len(im_size) == 3:
                result = image.mul(std.expand(im_size)).add(mean.expand(im_size))
            elif len(im_size) == 4:
                result = image.mul(std.unsqueeze(0).expand(im_size)).add(mean.unsqueeze(0).expand(im_size))
                
        return  result