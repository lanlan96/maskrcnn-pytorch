
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nms.pth_nms import pth_nms


def nms(dets, thresh):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  return pth_nms(dets, thresh)
