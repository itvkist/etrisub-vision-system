import configparser

class Config:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
    @property
    def num_cameras(self):
        return int(self.config['settings']['num_cameras'])
