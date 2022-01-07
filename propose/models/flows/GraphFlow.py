"""Basic definitions for the flows module."""

import torch.nn

import nflows.utils.typechecks as check

from nflows.flows.base import Flow

from torch_geometric.data import HeteroData


class GraphFlow(Flow):
    """
    Adaptation of Flow class for Hetero data.
    """

    def embed_inputs(self, inputs):
        """Embed the context inputs into the flow.

        Args:
            inputs: Tensor, input variables.

        Returns:
            A Tensor containing the embedded inputs, with shape [num_nodes, num_samples, ...].
        """
        inputs_dict = inputs.to_dict()

        if "c" not in inputs_dict:
            return inputs

        inputs_dict["c"]["x"] = self._embedding_net(inputs_dict["c"]["x"])

        return HeteroData(inputs_dict)

    def log_prob(self, inputs):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = self.embed_inputs(inputs)
        return self._log_prob(inputs)

    def _log_prob(self, inputs):
        noise, logabsdet = self._transform(inputs)

        log_prob = self._distribution.log_prob(noise["x"]["x"])

        return log_prob + logabsdet

    def sample(self, num_samples, context, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        context = self.embed_inputs(context)

        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        num_nodes = context["x"]["x"].shape[0] if context is not None else 1
        noise = self._distribution.sample((num_samples * num_nodes)).reshape(
            (num_nodes, num_samples, 3)
        )

        noise = self._noise_to_hetero(noise, context)

        samples, _ = self._transform.inverse(noise)

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        context = self.embed_inputs(context)

        num_nodes = context["x"]["x"].shape[0] if context is not None else 1

        noise, log_prob = self._distribution.sample_and_log_prob(
            (num_samples * num_nodes)
        )

        noise = noise.reshape((num_nodes, num_samples, 3))
        log_prob = log_prob.reshape((num_nodes, num_samples)).mean(-1)

        noise = self._noise_to_hetero(noise, context)

        samples, logabsdet = self._transform.inverse(noise)

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        inputs = self.embed_inputs(inputs)
        noise, _ = self._transform(inputs)
        return noise

    @staticmethod
    def _noise_to_hetero(noise, context):
        context_dict = context.to_dict()
        noise = {**context_dict, "x": {**context_dict["x"], "x": noise}}
        if "c" in noise:
            noise["c"]["x"] = noise["c"]["x"].unsqueeze(1)

        noise = HeteroData(noise)

        return noise

    @property
    def device(self):
        return next(self.parameters()).device
