import torchvision
import pdb
from torchsummary import summary


rn50_fcn = torchvision.models.segmentation.fcn_resnet50(False)
rn50 = torchvision.models.resnet50(False)
pdb.set_trace()