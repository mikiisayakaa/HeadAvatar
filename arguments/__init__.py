#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
import json

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])

        return group 

class MeshModelParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self.target_path = ""  # Path to the target data set for pose and expression transfer
        self.obj_path = ""
        self._model_path = ""
        self._vhap = False
        self.neural = False
        self.eval = True
        self.test = False
        self.specular_pbr = True
        self.differential = True
        self.train_light = True
        
        self.epoch = 3000
        self.check = 500
        self.semantic_epoch = [3000, self.epoch]
        self.texture_epoch = [0, 3000]
        self.vertex_epoch = [0, 3000]
        
        self.batch_size = 4
        self.batch_coeff = self.batch_size / 4.0
        
        self.position_lr = 1e-3 * self.batch_coeff
        
        self.albedo_lr = 0.01 * self.batch_coeff
        self.light_lr = 0.01 * self.batch_coeff
        self.roughness_lr = 0.001 * self.batch_coeff
        self.shader_lr = 0.01 * self.batch_coeff
        self.semantic_lr = 0.01 * self.batch_coeff
        
        self.lambda_l1 = 0.8
        self.lambda_dssim = 0.2
        self.lambda_smooth = 0.0
        self.lambda_mask = 20.0
        self.lambda_laplacian = 0.0
        self.lambda_roughness = 0.01
        self.lambda_semantic = 0.0
        self.lambda_semantic_reg = 0.1
        self.lambda_semantic_tv = 1.0
        self.lambda_lpips = 0.00
        self.lambda_shader_reg = 0.0
        
        self.texture_width = 1024
        self.texture_height = 1024
        
        self.train_ids = list(range(0, 16))
        self.val_ids = [8]
        self.train_ids = list(set(self.train_ids) - set(self.val_ids)) 
        self.timesteps = [0]
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelSequenceParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self.target_path = ""  # Path to the target data set for pose and expression transfer
        self.obj_path = ""
        self._model_path = ""
        self.eval = True
        self.test = False
        self._load_path = ""
        self._vhap = False
        self.neural = False
        self.differential = True
        self.specular_pbr = True
        self.train_light = True
        
        self.batch_size = 4
        self.epoch = 2500
        
        self.vertex_epoch = [0, 2000]
        self.texture_epoch = [0, 2500]
        
        self.batch_coeff = self.batch_size / 4.0
        self.position_lr = 0.001 * self.batch_coeff
        
        self.albedo_lr = 0.001 * self.batch_coeff
        self.light_lr = 0.01 * self.batch_coeff
        self.roughness_lr = 0.01 * self.batch_coeff
        self.semantic_lr = 0.01 * self.batch_coeff
        self.shader_lr = 0.001 * self.batch_coeff
        
        self.lambda_l1 = 0.8
        self.lambda_dssim = 0.2
        self.lambda_smooth = 0.0
        self.lambda_mask = 20.0
        self.lambda_laplacian = 0.0
        self.lambda_roughness = 0.01
        self.lambda_semantic = 40.0
        self.lambda_lpips = 0.01
        self.lambda_shader_reg = 0.0
        self.lambda_roughness_tv = 0.01
        
        self.texture_width = 1024
        self.texture_height = 1024
        
        self.train_ids = list(range(0, 16))
        # self.train_ids = [8]
        self.val_ids = [8]
        self.train_ids = list(set(self.train_ids) - set(self.val_ids))

        self.timesteps = [0, 14, 23, 86]
       
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelSequenceTexParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self.target_path = ""  # Path to the target data set for pose and expression transfer
        self._model_path = ""
        self.eval = True
        self.test = False
        self._vhap = False
        self._load_path = ""
        self.neural = False
        self.differential = False
        self.specular_pbr = True
        self.train_light = True
        
        self.batch_size = 4
        self.epoch = 30
        self.batch_coeff = self.batch_size / 4.0
        
        self.shader_lr = 0.001 * self.batch_coeff
        self.albedo_lr = 0.01 * self.batch_coeff
        self.light_lr = 0.01 * self.batch_coeff
        self.roughness_lr = 0.01 * self.batch_coeff
        
        self.lambda_l1 = 0.8
        self.lambda_dssim = 0.2
        self.lambda_roughness = 0.00
        self.lambda_shader_reg = 0.1
        self.lambda_lpips = 0.01
        self.lambda_albedo_tv = 0.0
        self.lambda_roughness_tv = 0.01
        
        self.texture_width = 1024
        self.texture_height = 1024
        self.output_resolution = 1
        
        self.train_ids = list(range(0, 16))
        # self.train_ids = [8]
        self.val_ids = [8]
        self.train_ids = list(set(self.train_ids) - set(self.val_ids))
        
        # self.timesteps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        # self.train_ids = [0, 3, 7, 11, 15]
        # self.val_ids = list(set(range(0, 16)) - set(self.train_ids))
        # self.train_ids = [0, 2, 4, 6, 7, 10, 12, 14, 15]
        # self.val_ids = list(set(range(0, 16)) - set(self.train_ids))
       
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelTestParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._load_path = ""
        self.test = True
        self._train = False
        self._vhap = False
        self.differential = True
        self.neural = False
        self.texture_width = 1024
        self.texture_height = 1024
        self.specular_pbr = True
        
        self.epoch = 200
        
        self.position_lr = 6e-3
        self.light_lr = 0.0
        
        self.lambda_l1 = 0.8
        self.lambda_dssim = 0.2
        self.lambda_mask = 20.0
        self.lambda_semantic = 400.0
        self.lambda_smooth = 0.0
        self.lambda_laplacian = 0.0
        self.lambda_lpips = 0.01
        
        self.test_train_ids = list(range(0, 16))
        self.test_val_ids = [8]
        # self.test_train_ids = list(set(self.test_train_ids) - set(self.test_val_ids))
        
        # self.timesteps = list(range(1402, 1531, 30))
        # self.timesteps = list(range(1466, 1555, 30))
        self.timesteps = list(range(0, 30))
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelTexEditParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._load_path = ""
        self.texture_width = 1024
        self.texture_height = 1024
        self.test = False
        self.differential = True
        self.target_color = [0.4, 0.0, 0.0]
        self.target_semantic = [7,9] # hair
        
        self.light_lr = 0.0
        self.position_lr = 0.0
        self.semantic_lr = 0.0
        self.epoch = 1

        self.train_ids= [7]
        self.val_ids = [8]
        self.timesteps = [0]
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelRelightParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._load_path = ""
        self._envmap_path = ""
        self.texture_width = 1024
        self.texture_height = 1024
        self.test = False
        self.differential = True
        
        self.light_lr = 0.0
        self.position_lr = 0.0
        self.semantic_lr = 0.0
        self.epoch = 1

        
        self.train_ids= [7]
        self.val_ids = [8]
        self.timesteps = [0]
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelExprEditParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._load_path = ""
        self.texture_width = 1024
        self.texture_height = 1024
        self.test = False
        self.differential = True
        
        self.light_lr = 0.0
        self.position_lr = 0.0
        self.semantic_lr = 0.0
        self.epoch = 1

        
        self.train_ids= [7]
        self.val_ids = [8]
        self.timesteps = [0]
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelExprTransferParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._target_path = ""  
        self._load_path = ""
        self._another_load_path = ""
        self.texture_width = 1024
        self.texture_height = 1024
        self.test = False
        self.differential = True
        
        self.light_lr = 0.0
        self.position_lr = 0.0
        self.semantic_lr = 0.0
        self.epoch = 1

        
        self.train_ids= [7]
        self.val_ids = [8]
        self.timesteps = [656]
        
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
class MeshModelIdentityTransferParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._target_path = ""  
        self._load_path = ""
        self._another_load_path = ""
        self.texture_width = 1024
        self.texture_height = 1024
        self.test = False
        self.differential = True
        
        self.light_lr = 0.0
        self.position_lr = 0.0
        self.semantic_lr = 0.0
        self.epoch = 1

        
        self.train_ids= [7]
        self.val_ids = [8]
        self.timesteps = [0]
        
        
        super().__init__(parser, "Mesh Model Parameters")
        
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
    
    

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

class TrainConfig():
    def __init__(self, op):
        self.op = op
        
    def save_json(self, path):
        json_data = {}
        json_data["MeshModelParams"] = self.op.__dict__
        
        with open(os.path.join(path, "config.json"), 'w') as fh:
            json.dump(json_data, fh)
            
    def load_json(self, path):
        with open(os.path.join(path, "config.json"), 'r') as fh:
            json_data = json.load(fh)
        
        op = MeshModelParams()
        for key, value in json_data["MeshModelParams"].items():
            setattr(op, key, value)
            
        return op
        
