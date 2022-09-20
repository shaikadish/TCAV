import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import alexnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelExplainer():
    """
    Wrapper for AlexNet, with capabilities to track layer activations and gradients

    Attributes
    ----------
    model : PyTorch model
        AlexNet model being wrapped by this class
    intermediate_activations : dict
        Layer activations for model. Indexed as intermediate_activations[LAYER NUMBER]
    layers : str
        Layers of model to be tracked

    Methods
    -------
    generate_gradients:
        Generate activation gradients for a given model layer

    """

    def __init__(self, layers,num_classes,load_path = None):
        
        # If using specific pretrained model, load from path
        if load_path:
          model = alexnet(num_classes = num_classes)
          model.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))
        # Load a model pretrained with ImageNet-1K
        else:
          model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)
          model.classifier[6] = nn.Linear(4096,num_classes)

        # Initialize internal variables
        self.model = model.to(device)
        self.intermediate_activations = {}
        self.layers = layers

        # Register forward hooks to track layer activations
        def save_activation(layer_name): def hook(model, input, output):
                self.intermediate_activations[layer_name] = output
            return hook

        for l in self.layers:
          self.model.features[l].register_forward_hook(save_activation(str(l)))

    def generate_gradients(self,layer_name,class_index):
    
      # Model activations for the desired layer
      acts = self.intermediate_activations[str(layer_name)]
    
      # Model outputs
      self.model.eval()
      outputs = self.output

      # Calculate gradients for specific class 
      grads = torch.autograd.grad(outputs[:,class_index], acts,grad_outputs=torch.ones_like(outputs[:, class_index]),create_graph=True)[0]
      
      return grads.detach().cpu().numpy()

    def __call__(self, x):
        self.output = self.model(x)
        return self.output
