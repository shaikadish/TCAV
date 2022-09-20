from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TCAV(object):
    """
    TCAV model, for producing TCAV scores between a given model layer and concept. Also contains CAV vectors for a set of concepts.

    Attributes
    ----------
    model : PyTorch model
        The model used to calculate CAV vectors
    cavs : dict
        Dictionary with CAV vectors for each concept and model layer. Indexed as cavs[CONCEPT][LAYER NUMBER]

    Methods
    -------
    get_sensitivites:
        Calculates the dot product between a CAV vector and the gradients of a given model layer. These sensitivites are used in calculating the TCAV score
    get_score:
        Calculates the TCAV score by calculating the ratio of sensitivites which are positive for a given layer and concept
    get_sensitive_indexes:
        Generates indexes of dataset which have a positive sensitivity to a given CAV vector
    get_CAV:
        Generates the CAV vector for an input concept dataset
    """

    def __init__(self, model, concept_datasets):

        # Save Model
        self.model = model
        self.model.model.eval()

        # Generate CAV for each concept
        self.cavs = {}
        for concept_dataset in concept_datasets:
            cav, acc = self.get_CAV(concept_dataset)
            self.cavs[concept_dataset.datasets[0].root] = cav

    def get_sensitivities(self, gradients, concept, data_class, layer):
        return np.dot(gradients[data_class][layer].reshape(gradients[data_class][layer].shape[0], -1), self.cavs[concept][layer])

    def get_score(self, gradients, concept, data_class, layer):
        dots = self.get_sensitivities(gradients, concept, data_class, layer)
        return len(dots[dots > 0])/len(dots)

    def get_sensitive_indexes(self, gradients, concept, data_class, layer):
        dots = self.get_sensitivities(gradients, concept, data_class, layer)
        return np.where(np.array(dots) > 0)[0]

    def get_CAV(self, concept_dataset):

        # Generate model activations for concept dataset
        linear_loader = torch.utils.data.DataLoader(
            concept_dataset, batch_size=len(concept_dataset))
        data, activations_targets = next(iter(linear_loader))
        self.model(data.to(device))

        accs = {}
        cavs = {}

        # Generate CAV vector for each model layer
        for layer in self.model.intermediate_activations:

            # Extract model layer activations
            activations_features = self.model.intermediate_activations[layer].cpu(
            ).detach().numpy()
            
            activations_features = (activations_features-activations_features.mean())/activations_features.std()

            # Train linear model
            x_train, x_test, y_train, y_test = train_test_split(
                activations_features.reshape((activations_features.shape[0], -1)), activations_targets, test_size=0.15,random_state=42)
        
            lm = linear_model.SGDClassifier(alpha=0.01, max_iter=2000, tol=1e-3,random_state=42)
            lm.fit(x_train, y_train)

            # Calculate linear model prediction accuracy
            y_pred = lm.predict(x_test)
            num_classes = len(activations_targets.unique())
            acc = {}
            num_correct = 0
            for class_id in range(num_classes):
                idx = (y_test.numpy() == np.array(class_id))
                acc[str(class_id)] = metrics.accuracy_score(
                    y_pred[idx], np.array(y_test)[idx])
                num_correct += (sum(idx) * acc[str(class_id)])
            acc['overall'] = float(num_correct) / float(len(y_test))

            # Store CAV and linear model accutacy
            accs[layer] = acc
            cavs[layer] = lm.coef_[0]

        return cavs, accs
