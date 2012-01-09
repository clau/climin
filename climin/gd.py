# -*- coding: utf-8 -*-

import itertools


from base import Minimizer, repeat_or_iter


class GradientDescent(Minimizer):

    def __init__(self, wrt, fandprime, steprate, momentum=0.0, 
                 args=None, stop=1, verbose=False):
        super(GradientDescent, self).__init__(
            wrt, args=args, stop=stop, verbose=verbose)

        self.steprates = repeat_or_iter(steprate)
        self.momentums = repeat_or_iter(momentum)
        self.fandprime = fandprime

    def __iter__(self):
        step_m1 = 0
        periterargs = itertools.izip(self.steprates, self.momentums, self.args)
        for i, j in enumerate(periterargs):
            steprate, momentum, (args, kwargs) = j
            loss, gradient = self.fandprime(self.wrt, *args, **kwargs)
            step = gradient * steprate + momentum * step_m1
            self.wrt -= step

            if (i + 1) % self.stop == 0:
                yield dict(
                    loss=loss, gradient=gradient, steprate=steprate, 
                    momentum=momentum, step=step, wrt=self.wrt)

            step_m1 = step
