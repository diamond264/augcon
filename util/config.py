import yaml

class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        for key, value in config_dict.items():
            self.set_config(key, value)

    def set_config(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                getattr(self, key).set_config(k, v)
        else:
            setattr(self, key, value)
    
    def log_config(self, logger, prefix=''):
        configs = vars(self)
        for key in configs.keys():
            val = configs[key]
            if isinstance(val, Config):
                logger.info(f'  {prefix}{key}:')
                val.log_config(logger, prefix+'  ')
            else:
                logger.info(f'  {prefix}{key}: {configs[key]}')