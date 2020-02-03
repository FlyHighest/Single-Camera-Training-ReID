from __future__ import print_function, absolute_import
from datasets.raw_data_loader import RawImageData
from datasets.data_components.market_sct import MarketSCT

from datasets.data_components.duke_sct import DukeSCT
"""Create data_components"""

__data_factory = {
    'market_sct': MarketSCT,
    'duke_sct':DukeSCT,
}

__folder_factory = {
    'market_sct': RawImageData,
    'duke_sct': RawImageData
}


def init_dataset(name, *args, **kwargs):
    if name not in __data_factory.keys():
        raise KeyError("Unknown data_components: {}".format(name))
    return __data_factory[name](*args, **kwargs)


def init_datafolder(name, data_list, transforms, if_train):
    if name not in __folder_factory.keys():
        raise KeyError("Unknown data_components: {}".format(name))
    raw= __folder_factory[name](data_list, transforms, if_train)
    return raw
