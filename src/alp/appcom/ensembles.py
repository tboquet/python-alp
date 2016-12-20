from time import time

import numpy as np
import pandas as pd
from progressbar import ETA
from progressbar import Bar
from progressbar import DynamicMessage
from progressbar import FormatLabel
from progressbar import Percentage
from progressbar import ProgressBar
from progressbar import SimpleProgress


def get_ops(metric):
    if metric in ['loss', 'val_loss']:
        op = np.min
        op_arg = np.argmin
        is_max = False
    else:
        op = np.max
        op_arg = np.argmax
        is_max = True
    return op, op_arg, is_max


def get_best(experiments, metric):
    op, op_arg, is_max = get_ops(metric)
    best_perf_expes = []
    for expe in experiments:
        if not hasattr(expe, 'full_res'):
            raise Exception('Results are not ready')
        best_perf_expes.append(op(expe.full_res['metrics'][metric]))
    return experiments[op_arg(best_perf_expes)]


widgets = [Percentage(), ' ',
           SimpleProgress(), ' ',
           Bar(marker='=', left='[', right=']'),
           ' ', FormatLabel('in: %(elapsed)s'), ' ',
           ETA(), ' | ', 'job/', DynamicMessage('s')]


class Ensemble(object):

    def __init__(self, experiments):
        self.experiments = experiments

    def fit(self, data, data_val, *args, **kwargs):
        raise NotImplementedError

    def fit_gen(self, data, data_val, *args, **kwargs):
        raise NotImplementedError

    def fit_async(self, data, data_val, *args, **kwargs):
        raise NotImplementedError

    def fit_gen_async(self, data, data_val, *args, **kwargs):
        raise NotImplementedError

    def predict(self, data, data_val, *args, **kwargs):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def plt_summary(self, verbose=False):
        raise NotImplementedError


class HParamsSearch(Ensemble):
    """Hyper parameters search class

    Train several experiments with different hyperparameters and save results.
    Wraps the training process so that it's possible to access results easily.

    Args:
        experiments(list): a list of experiments
        hyperparams(dict): a dict of hyperparameters
        metric(str): the name of a metric used in the experiments

    """
    def __init__(self, experiments, hyperparams=None, metric=None):
        super(HParamsSearch, self).__init__(experiments=experiments)
        self.hyperparams = hyperparams
        self.metric = metric
        self.results = []

    def fit(self, data, data_val, *args, **kwargs):
        """Apply the fit method to all the experiments

        Args:
            see `alp.core.Experiment.fit`

        Returns:
            a list of results"""
        self._fit_cm(data, data_val, gen=False, async=False, *args, **kwargs)
        return self.results

    def fit_gen(self, data, data_val, *args, **kwargs):
        """Apply the fit_gen method to all the experiments

        Args:
            see :meth:`alp.appcom.core.Experiment.fit_gen`

        Returns:
            a list of results"""
        self._fit_cm(data, data_val, gen=True, async=False, *args, **kwargs)
        return self.results

    def fit_gen_async(self, data, data_val, *args, **kwargs):
        """Apply the fit_gen_async method to all the experiments

        Args:
            see :meth:`alp.appcom.core.Experiment.fit_gen_async`

        Returns:
            a list of results"""
        self._fit_cm(data, data_val, gen=True, async=True, *args, **kwargs)
        return self.results

    def fit_async(self, data, data_val, *args, **kwargs):
        """Apply the fit_async method to all the experiments

        Args:
            see :meth:`alp.appcom.core.Experiment.fit_async`

        Returns:
            a list of results"""
        self._fit_cm(data, data_val, gen=False, async=True, *args, **kwargs)
        return self.results

    def _fit_cm(self, data, data_val, gen, async, *args, **kwargs):
        with ProgressBar(max_value=len(self.experiments),
                         redirect_stdout=True,
                         widgets=widgets, term_width=80) as progress:
            for i, expe in enumerate(self.experiments):
                b = time()
                if gen and async:
                    res = expe.fit_gen_async(data, data_val, *args, **kwargs)
                elif gen and not async:
                    res = expe.fit_gen(data, data_val, *args, **kwargs)
                elif not gen and async:
                    res = expe.fit_async(data, data_val, *args, **kwargs)
                else:
                    res = expe.fit(data, data_val, *args, **kwargs)

                self.results.append(res)
                if i == 0:
                    spent = time() - b
                    to_print = spent
                else:
                    spent += time() - b
                    to_print = spent / (i + 1)
                progress.update(i, s=float(1 / to_print))
                if expe.backend == 'keras' and async:  # pragma: no cover
                    import keras.backend as K
                    if K.backend() == 'tensorflow':
                        K.clear_session()
        return self.results

    def predict(self, data, *args, **kwargs):
        """Apply the predict method to all the experiments

        Args:
            see :meth:`alp.appcom.core.Experiment.predict`

        Returns:
            an array of results"""
        if self.metric is None:
            self.metric = 'loss'
        best = get_best(self.experiments, self.metric)
        return best.predict(data, *args, **kwargs)

    def summary(self, verbose=False):
        """Build a results table using individual results from models

        Args:
            verbose(bool): if True, print a description of the results

        Returns:
            a pandas DataFrame of results"""
        # build results table
        res_dict = dict()
        expes = self.experiments
        for i, res in enumerate(self.results):
            res, t = res
            if t is not None:
                t.join()
            for k, v in expes[i].full_res['metrics'].items():
                if isinstance(v, list):
                    op, _, _ = get_ops(k)
                    if k in res_dict:
                        res_dict[k] += [op(v)]
                    else:
                        res_dict[k] = []
                        res_dict[k] += [op(v)]
        res_table = pd.DataFrame(res_dict)
        if verbose is True:
            print(res_table.describe())
        return res_table
