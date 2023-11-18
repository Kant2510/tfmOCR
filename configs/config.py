import yaml

class Config(dict):
    def __init__(self, config_dict):
        super(Config, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config(fname='default.yml'):
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return Config(config)

    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True, encoding='utf-8')

