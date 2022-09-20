import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor(object):
    """
    For extracting batches of activations or gradients for a model, using a set of inputs

    Attributes
    ----------
    model : PyTorch model
        Model whose activations/gradients are extracted

    Methods
    -------
    get_activations:
        Generate all layer activations of the model for a given dataset
    get_gradients
        Generate all layer gradients of the model for a given dataset
    get_act_grad:
        Generate all layer activations and gradients of the model for a given dataset
    get_feature_dicts:
        Function to be used privately. Calculates activations/gradients in batches using input dataset

    """

    def __init__(self, model):
        self.model = model

    def get_activations(self, dataset):
        self.model.model.eval()
        return self.get_feature_dicts(
            dataset, activations=True, gradients=False)

    def get_gradients(self, dataset):
        self.model.model.eval()
        return self.get_feature_dicts(
            dataset, activations=False, gradients=True)

    def get_act_grad(self, dataset):
        self.model.model.eval()
        return self.get_feature_dicts(
            dataset, activations=True, gradients=True)

    def get_feature_dicts(self, dataset, activations=True, gradients=False):

        # Track activations/gradients for each data class
        class_dict = {}

        for c in np.unique(dataset.targets):

            # Create subset for class c
            sampler = np.where(np.array(dataset.targets) == c)
            class_loader = torch.utils.data.DataLoader(
                dataset, batch_size=16, sampler=sampler[0])

            # Initialize dicts
            grads_dict = {}
            activations_dict = {}

            for l in self.model.layers:
                grads_dict[str(l)] = None
                activations_dict[str(l)] = None

            # Extract model activations and gradients for data subset
            for data in class_loader:
                d, t = data

                # Update model internal activations
                self.model(d.to(device))

                # Store activations and gradients for each tracked model layer
                for l in self.model.layers:
                    if (activations_dict[str(l)] is None) and (
                            grads_dict[str(l)] is None):
                        if activations:
                            activations_dict[str(l)] = self.model.intermediate_activations[str(
                                l)].cpu().detach().numpy()
                        if gradients:
                            grads_dict[str(l)] = self.model.generate_gradients(
                                l, t[0].item())
                    else:
                        if activations:
                            activations_dict[str(l)] = np.concatenate((activations_dict[str(
                                l)], self.model.intermediate_activations[str(l)].cpu().detach().numpy()), axis=0)
                        if gradients:
                            grads_dict[str(l)] = np.concatenate(
                                (grads_dict[str(l)], self.model.generate_gradients(l, t[0])), axis=0)

            # Define output dicts
            if activations and gradients:
                class_dict[str(c)] = {
                    'activations': activations_dict, 'gradients': grads_dict}

            elif activations:
                class_dict[str(c)] = activations_dict

            elif gradients:
                class_dict[str(c)] = grads_dict

        return class_dict
