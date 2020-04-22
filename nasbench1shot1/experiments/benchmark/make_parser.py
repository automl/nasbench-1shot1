import yaml
import argparse

def make_parser(existing_parser=None, **kwargs):
    if existing_parser:
        parser = existing_parser
    else:
        parser = argparse.ArgumentParser(**kwargs)


def config_reader(config_file='default.yaml'):
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise(exc)
        else:
            return AttrDict(config)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
