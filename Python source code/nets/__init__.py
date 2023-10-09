from .mobilenet import mobilenet_v2
#from .mobilenet import ModifiedMobileNetV2
get_model_from_name = {
   "mobilenet": mobilenet_v2,
  # "mobilenet":  ModifiedMobileNetV2,
}