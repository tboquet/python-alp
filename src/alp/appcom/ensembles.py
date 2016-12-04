from progressbar import ProgressBar
from progressbar import Bar
from progressbar import DynamicMessage
from progressbar import ETA
from progressbar import FormatLabel
from progressbar import Percentage
from progressbar import SimpleProgress
from time import time
import numpy as np


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
    best_perf_expes = [op(expe.full_res['metrics'][metric])
                       for expe in experiments]
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

    def plt_summary(self):
        raise NotImplementedError


class HParamsSearch(Ensemble):
    """Hyper parameters search class
    """
    def __init__(self, experiments, hyperparams=None, metric=None):
        super(HParamsSearch, self).__init__(experiments=experiments)
        self.hyperparams = hyperparams
        self.metric = metric
        self.results = []

    def fit(self, data, data_val, *args, **kwargs):
        for expe in experiments:
            res = expe.fit(data, data_val, *args, **kwargs)
            self.results.append(res)
        return self.results

    def fit_gen(self, data, data_val, *args, **kwargs):
        with ProgressBar(max_value=len(self.experiments),
                         redirect_stdout=True,
                         widgets=widgets, term_width=80) as progress:
            for i, expe in enumerate(self.experiments):
                b = time()
                res = expe.fit_gen(data, data_val, *args, **kwargs)
                self.results.append(res)
                if i == 0:
                    spent = time() - b
                else:
                    spent += time() - b
                    spent /= i + 1
                progress.update(i, s=float(1/spent))
                if expe.backend == 'keras':
                    import keras.backend as K
                    if K.backend() == 'tensorflow':
                        K.clear_session()
        return self.results

    def fit__gen_async(self, data, data_val, *args, **kwargs):
        with ProgressBar(max_value=len(self.experiments),
                         redirect_stdout=True,
                         widgets=widgets, term_width=80) as progress:
            for i, expe in enumerate(self.experiments):
                b = time()
                res = expe.fit_gen_async(data, data_val, *args, **kwargs)
                self.results.append(res)
                if i == 0:
                    spent = time() - b
                else:
                    spent += time() - b
                    spent /= i + 1
                progress.update(i, s=float(1/spent))
                if expe.backend == 'keras':
                    import keras.backend as K
                    if K.backend() == 'tensorflow':
                        K.clear_session()
        return self.results

    def fit_async(self, data, data_val, *args, **kwargs):
        with ProgressBar(max_value=len(self.experiments),
                         redirect_stdout=True,
                         widgets=widgets, term_width=80) as progress:
            for i, expe in enumerate(self.experiments):
                b = time()
                res = expe.fit_async(data, data_val, *args, **kwargs)
                self.results.append(res)
                if i == 0:
                    spent = time() - b
                else:
                    spent += time() - b
                    spent /= i + 1
                progress.update(i, s=float(1/spent))
                if expe.backend == 'keras':
                    import keras.backend as K
                    if K.backend() == 'tensorflow':
                        K.clear_session()
        return self.results

    def predict(self, data, *args, **kwargs):
        if self.metric is None:
            self.metric = 'loss'
        best = get_best(self.experiments, self.metric)
        return best.predict(data, *args, **kwargs)

    def summary(self, verbose=False):
        # build results table
        res_dict = dict()
        expes = self.experiments
        for i, res in enumerate(self.results):
            res, t = res
            t.join()
            for k, v in expes[i].full_res['metrics'].items():
                if isinstance(v, list):
                    if k in min_metric:
                        op = np.min
                    else:
                        op = np.max
                    if k in res_dict:
                        res_dict[k] += [op(v)]
                    else:
                        res_dict[k] = []
                        res_dict[k] += [op(v)]
        res_table = pd.DataFrame(res_dict)
        if verbose is True:
            print(res_table.describe())
        return res_table
