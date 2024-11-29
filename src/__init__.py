import torch
import transformers

# set verbosity to error to avoid unnecessary logs from torch and transformers
torch.set_printoptions(profile="short")
transformers.utils.logging.set_verbosity_error()