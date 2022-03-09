import torchvision
import pdb
from torchsummary import summary


model = torchvision.models.segmentation.fcn_resnet50(False)
rn34 = torchvision.models.resnet50(False)
pdb.set_trace()