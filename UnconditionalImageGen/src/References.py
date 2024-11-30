import sys
sys.path.append("./src/")

class DatasetRef:
    def __init__(self, name, url, row):
        self.name = name
        self.url = url
        self.row = row

class ModelRef:
    def __init__(self, name, url, unet_classname, scheduler_classname):
        self.name = name
        self.url = url
        self.unet_classname = unet_classname
        self.scheduler_classname = scheduler_classname