# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:55:02 2025

@author: bhuyn
"""
from scipy.stats import linregress

class LinReg:
    def __init__(self, x, y, xlabel):
        self.x = x
        self.xlabel = xlabel # q or w
        self.y = y
        self.m, self.int, self.r, self.p, self.std_err = linregress(x, y)