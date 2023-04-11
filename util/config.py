
class Config:
    def __init__(self):
        self.seed = None
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.data = DataConfig()
        self.multiprocessing = MultiprocessingConfig()

class ModelConfig:
    def __init__(self):
        self.type = None
        self.num_layers = None
        self.hidden_size = None

class TrainConfig:
    def __init__(self):
        self.resume = None
        self.epochs = None
        self.start_epoch = None
        self.batch_size = None
        self.lr = None
        self.momentum = None
        self.weight_decay = None
        self.print_freq = None
        self.save_dir = None

class DataConfig:
    def __init__(self):
        self.dir = None

class MultiprocessingConfig:
    def __init__(self):
        self.multiprocessing_distributed = None
        self.workers = None
        self.sworld_size = None
        self.rank = None
        self.dist_url = None
        self.dist_backend = None
        self.gpu = None