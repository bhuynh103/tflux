# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 08:36:49 2025

@author: bhuyn
"""

import numpy as np
from grid import Grid
from mesh import Mesh
from linreg import LinReg


class Junction:
    def __init__(self, vertices, is_top, filename):
        self.filename = filename
        self.vertices = vertices
        self.is_top = is_top
        self.grid: Grid = None
        self.mesh: Mesh = None
        self.linreg_q: LinReg = None
        self.linreg_w: LinReg = None


class Sample:
    def __init__(self):
        self.juncs = []
    
    
    def append_junction(self, junc):
        self.juncs.append(junc)
        return self
    
    
    def find_average(self, attr):
        match attr:
            case 'a':
                attr_list = [junc.mesh.a for junc in self.juncs]
            case 'b':
                attr_list = [junc.mesh.b for junc in self.juncs]
            case 'q_m':
                attr_list = [junc.linreg_q.m for junc in self.juncs]
            case 'w_m':
                attr_list = [junc.linreg_w.m for junc in self.juncs]
        
        mean = np.mean(attr_list)
        std = np.std(attr_list)
        return mean, std