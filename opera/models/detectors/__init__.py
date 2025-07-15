# Copyright (c) Hikvision Research Institute. All rights reserved.
from .dkdetr import DKDETR
from .inspose import InsPose
from .petr import PETR
from .soit import SOIT
from .videoposev1 import VideoPoseV1
from .videoposev2 import VideoPoseV2

__all__ = ['DKDETR', 'InsPose', 'PETR', 'SOIT', 'VideoPoseV1', 'VideoPoseV2']
