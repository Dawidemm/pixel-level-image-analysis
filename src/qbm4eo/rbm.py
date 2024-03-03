import abc
from itertools import islice

import dimod
import numpy as np
from scipy.special import expit
from torch.utils.data import DataLoader
import tqdm


INITIAL_COEF_SCALE = 0.1


def infinite_dataloader_generator(data_loader):
    while True:
        for batch_idx, (data, target) in enumerate(data_loader):
            yield batch_idx, (data, target)


def qubo_from_rbm_coefficients(
    weights: np.ndarray, v_bias: np.ndarray, h_bias: np.ndarray
):
    """Create a QUBO problem representing RBM with given coefficients.

    :param weights: a NxM interaction matrix.
    :param v_bias: a N-element visible layer bias vector.
    :param h_bias: a M-element hidden layer bias vector.
    :return: A QUBO, represented as dimod.BQM with N+M variables, such that:
     - variables 0,...,N-1 correspond to hidden layer
     - variables 0,...,M-1 correspond to visible layer
     - visible layer biases correspond to linear coefficients of first N variables
     - hidden layer biases correspond to linear coefficients of second M variables
     - weights correspond to interaction terms between first N and second M variables.

     .. note::
        This function does not allow for manipulating how RBM variables are mapped to
        the QUBO variables. This is not a problem if QUBO is to be used with a unstructured
        sampler. For sampler, the intended usage is to wrap it in the EmbeddingComposite.
    """
    linear = {
        **{i: float(bias) for i, bias in enumerate(v_bias)},
        **{i: float(bias) for i, bias in enumerate(h_bias, start=len(v_bias))},
    }
    quadratic = {
        (i, j + len(v_bias)): float(weights[i, j])
        for i in range(len(v_bias))
        for j in range(len(h_bias))
    }

    return dimod.BQM(linear, quadratic, offset=0, vartype="BINARY")


class RBM:
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        rng=None
    ):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.rng = rng if rng is not None else np.random.default_rng()

        self.weights = (
            self.rng.normal(size=(self.num_visible, self.num_hidden)) * INITIAL_COEF_SCALE
        )
        self.v_bias = self.rng.normal(size=self.num_visible) * INITIAL_COEF_SCALE
        self.h_bias = self.rng.normal(size=self.num_hidden) * INITIAL_COEF_SCALE

    def h_probabilities_given_v(self, v_batch):
        return expit(self.h_bias + v_batch @ self.weights)

    def sample_h_given_v(self, v_batch):
        probabilities = self.h_probabilities_given_v(v_batch)
        return (self.rng.random(probabilities.shape) < probabilities).astype(float)

    def v_probabilities_given_h(self, h_batch):
        return expit(self.v_bias + h_batch @ self.weights.T)

    def sample_v_given_h(self, h_batch):
        probabilities = self.v_probabilities_given_h(h_batch)
        return (self.rng.random(probabilities.shape) < probabilities).astype(float)

    def reconstruct(self, v_batch):
        return self.v_probabilities_given_h(self.h_probabilities_given_v(v_batch))

    def save(self, file):
        np.savez(file, v_bias=self.v_bias, h_bias=self.h_bias, weights=self.weights)

    @classmethod
    def load(cls, file):
        data = np.load(file)
        weights, v_bias, h_bias = data["weights"], data["v_bias"], data["h_bias"]
        rbm = cls(num_visible=len(v_bias), num_hidden=len(h_bias))
        rbm.weights = weights
        rbm.v_bias = v_bias
        rbm.h_bias = h_bias
        return rbm


class RBMTrainer:

    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def fit(self, rbm: RBM, data_loader: DataLoader, callback=None):
        for i, (_idx, (batch, target)) in enumerate(pbar := tqdm.tqdm(islice(
                infinite_dataloader_generator(data_loader),
                self.num_steps
        ), total=self.num_steps)):
            batch = batch.detach().cpu().numpy().squeeze()
            self.training_step(rbm, batch)
            loss = ((batch-rbm.reconstruct(batch)) ** 2).sum() / batch.shape[0] / batch.shape[1]
            pbar.set_postfix(loss=loss)
            if callback is not None:
                callback(i, rbm, loss)

    @abc.abstractmethod
    def training_step(self, rbm: RBM, batch):
        pass


class AnnealingRBMTrainer(RBMTrainer):

    def __init__(
        self,
        num_steps: int,
        sampler: dimod.Sampler,
        qubo_scale: float = 1.0,
        learning_rate: float = 0.01,
        **sampler_kwargs
    ):
        super().__init__(num_steps)
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs
        self.qubo_scale = qubo_scale
        self.learning_rate = learning_rate

    def training_step(self, rbm, batch):
        # Conditional probabilities given visible batch input
        hidden = rbm.h_probabilities_given_v(batch)
        # Construct QUBO from this RBM
        bqm = qubo_from_rbm_coefficients(rbm.weights, rbm.v_bias, rbm.h_bias)
        # Scaling to compensate the temperature difference. Strangely, it seems
        # that in dimod this operation has to be done in place.
        bqm.scale(self.qubo_scale)
        # Take a sample of the same size as batch, extract only visible and hidden variables
        if "num_reads" in self.sampler.parameters:
            sample = self.sampler.sample(
                bqm, num_reads=len(batch), **self.sampler_kwargs
            ).record["sample"]
        else:
            sample = dimod.concatenate(
                [self.sampler.sample(bqm, **self.sampler_kwargs) for _ in range(len(batch))]
            ).record["sample"]
        # Split, remembering that first variables correspond to hidden layer
        sample_v = sample[:, :rbm.num_visible]
        sample_h = sample[:, rbm.num_visible:]
        # Update weights
        rbm.weights += (
            self.learning_rate * (batch.T @ hidden - sample_v.T @ sample_h) / len(batch)
        )
        # And biases
        rbm.v_bias += self.learning_rate * (batch - sample_v).sum(axis=0)
        rbm.h_bias += self.learning_rate * (hidden - sample_h).sum(axis=0)


class CD1Trainer(RBMTrainer):

    def __init__(self, num_steps: int, learning_rate: float = 0.01):
        super().__init__(num_steps)
        self.learning_rate = learning_rate

    def training_step(self, rbm: RBM, batch):
        # Conditional probabilities given visible batch input
        hidden_1 = rbm.h_probabilities_given_v(batch)

        # Propagate hidden -> visible -> hidden again
        visible_2 = rbm.v_probabilities_given_h(hidden_1)
        hidden_2 = rbm.h_probabilities_given_v(visible_2)

        # Update weights
        rbm.weights += (
                self.learning_rate * (batch.T @ hidden_1 - visible_2.T @ hidden_2) / len(batch)
        )
        # And biases
        rbm.v_bias += self.learning_rate * (batch - visible_2).sum(axis=0)
        rbm.h_bias += self.learning_rate * (hidden_1 - hidden_2).sum(axis=0)