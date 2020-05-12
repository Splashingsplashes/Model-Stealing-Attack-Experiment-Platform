#!/usr/bin/python

import knockoff
import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet
import code


# modelname = 'resnet34'
# model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
print(knockoff.models.imagenet.__dict__.keys())
