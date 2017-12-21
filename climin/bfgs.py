# -*- coding: utf-8 -*-

"""This module provides an implementation of Quasi-Newton methods
(BFGS, sBFGS and l-BFGS).

The Taylor expansion up to second order of a function :math:`f(\\theta_t)`
allows a local quadratic approximiation of :math:`f(\\theta_t + d_t)`:

    .. math::
         f(\\theta_t + d_t) \\approx f(\\theta_t) + d_t^Tf'(\\theta_t) + \\frac{1}{2}d_t^TH_td_t

where the symmetric positive definite matrix :math:`H_t` is the Hessian at :math:`\\theta_t`.
The minimizer :math:`d_t` of this convex quadratic model is:

    .. math::
        d_t = -H^{-1}f'(\\theta_t).

For large scale problems both computing/storing the Hessian and solving the above linear
system is computationally demanding. Instead of recomputing the Hessian from scratch at every
iteration, quasi-Newton methods utilize successive measurements of the gradient
to build a sufficiently good quadratic model of the objective function. The above formula
is then applied to yield a direction :math:`d_t`. The update done is then of the form

    .. math::
        \\theta_{t+1} = \\alpha_t d_t + \\theta_t

where :math:`\\alpha_t` is obtained with a line search.

.. note::
    The classes presented here are not working with gnumpy.


"""

from __future__ import absolute_import

import warnings

import scipy
import numpy as np
import scipy.linalg
import scipy.optimize

from .base import Minimizer, is_nonzerofinite
from .linesearch import WolfeLineSearch

def lbfgs_two_loop_recursion(grad, s, y, idxs):
    if len(idxs) == 0:
        return -grad
    else:
        q = grad.copy()
        n_current_factors = len(idxs)
        rk = scipy.empty(n_current_factors)
        a = scipy.empty(n_current_factors)

        for i in idxs[::-1]:
            rk[i] = 1 / scipy.inner(y[i], s[i])
            a[i] = rk[i] * scipy.inner(s[i], q)
            q -= a[i] * y[i]

        e = idxs[-1]
        Hk0 = scipy.inner(s[e], y[e]) / scipy.inner(y[e], y[e])
        R = Hk0 * q

        for i in idxs:
            beta = rk[i] * scipy.inner(y[i], R)
            R += s[i] * (a[i] - beta)

        return -R

class QuasiNewton(object):
    def __init__(self, memory=10, mode='BB'):
        self.memory = memory
        self.mode = mode
        self.adagrad_sum = 0
        self.rms_sum = 0
        self.reset()
        self.sqrt_eps = scipy.power(np.finfo(float).eps, 0.5)

    def reset(self):
        self.S = []
        self.Y = []

    def store(self, s, y):
        self.S.append(s)
        self.Y.append(y)
        if len(self.S) > self.memory:
            self.S.pop(0)
            self.Y.pop(0)

    def two_loop(self, g):
        if self.mode != 'BB':
            g2 =  scipy.square(g)
            self.adagrad_sum += g2
            self.rms_sum = self.rms_sum * 0.9 + 0.1 * g2

        if self.mode == 'BB':
            if len(self.S) == 0:
                return g
            Hk0 = scipy.dot(self.S[-1], self.Y[-1]) / scipy.dot(self.Y[-1], self.Y[-1])
        elif self.mode == 'ADAGRAD':
            Hk0 = 1 / scipy.power(self.adagrad_sum + self.sqrt_eps, 0.5)
        elif self.mode == 'RMS':
            Hk0 = 1 / scipy.power(self.rms_sum + self.sqrt_eps, 0.5)

        n = len(self.S)
        idxs = range(n)

        rk = scipy.empty(n)
        a = scipy.empty(n)
        q = g.copy()

        for i in reversed(idxs):
            rk[i] = 1 / scipy.dot(self.Y[i], self.S[i])
            a[i] = rk[i] * scipy.dot(self.S[i], q)
            q -= a[i] * self.Y[i]

        R = Hk0 * q

        for i in idxs:
            beta = rk[i] * scipy.dot(self.Y[i], R)
            R += self.S[i] * (a[i] - beta)

        return R

class AdaQn(Minimizer):

    def __init__(self, wrt, f, fprime,
                 alpha=0.0086, L=16, fisher_memory=100, lbfgs_memory=20, 
                 line_search=None, args=None):
        super(AdaQn, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime
        
        self.alpha = alpha
        self.L = L
        self.fisher_memory = fisher_memory
        self.lbfgs_memory = lbfgs_memory

        # if line_search is not None:
        #     self.line_search = line_search
        # else:
        #     self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')

    def compute_y(self, F, s):
        return np.dot(F, np.dot(F.T, s)) / F.shape[1]

    def __iter__(self):
        k = 1
        t = -1

        dim = self.wrt.shape
        w_s = scipy.zeros(dim)
        w_o = scipy.zeros(dim)

        qn = QuasiNewton(memory=self.lbfgs_memory)
        fisher_container = None
        
        args1, kwargs1 = next(self.args)
        
        # sg = self.fprime(self.wrt, *args1, **kwargs1)
        L = self.L

        for i, (args, kwargs) in enumerate(self.args):
            sg = self.fprime(self.wrt, *args, **kwargs)

            sg_ = sg.copy()
            if fisher_container is None:
                fisher_container = sg_[:,np.newaxis]
            else:
                fisher_container = scipy.column_stack((fisher_container, sg_[:,np.newaxis]))
                if fisher_container.shape[1] > self.fisher_memory:
                    fisher_container = scipy.delete(fisher_container, 0, axis=1)

            w_s += self.wrt

            if k < 1000:
                self.wrt -= self.alpha * qn.two_loop(sg)
                if k % L == 0:
                    t += 1
                    w_n = w_s / L
                    w_s = scipy.zeros(dim)
                    if t > 0:
                        if self.f(w_n, *args1) > 1.01 * self.f(w_o, *args1):
                            qn.reset()
                            self.wrt = w_o
                            fisher_container = None
                        else:
                            s = w_n - w_o
                            y = self.compute_y(fisher_container, s)
                            rho = np.dot(s, y) / np.dot(y, y)
                            if rho > 1e-4:
                                qn.store(s, y)
                                w_o = w_n
                    else:
                        w_o = w_n
            
            k += 1

            info = {
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
            }
            yield info

class Ssvm(Minimizer):
    """SSVM

    Attributes
    ----------
    wrt : array_like
        Current solution to the problem. Can be given as a first argument to \
        ``.f`` and ``.fprime``.

    f : Callable
        The object function.

    fprime : Callable
        First derivative of the objective function. Returns an array of the \
        same shape as ``.wrt``.

    initial_inv_hessian : array_like
        The initial estimate of the approximiate Hessian.

    line_search : LineSearch object.
        Line search object to perform line searches with.

    args : iterable
        Iterator over arguments which ``fprime`` will be called with.

    """

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None):
        """Create a BFGS object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``f`` and ``fprime`` should accept this array as a first argument.

        f : callable
            The objective function.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        initial_inv_hessian : array_like
            The initial estimate of the approximiate Hessian.

        line_search : LineSearch object.
            Line search object to perform line searches with.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        super(Ssvm, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)


    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')

    def find_direction(self, grad_m1, grad, step, inv_hessian):
        H = self.inv_hessian
        g = grad - grad_m1
        x = step

        # TODO, also line search
        phi = 0.1
        theta = 0.1

        # s'g = np.inner(s, g)
        # Hg = np.dot(H, g)
        # ss' = np.outer(s, s)
        xx = np.outer(x, x)
        xg = np.dot(x.T, g)

        Hg = np.dot(H, g) # grad-shape
        gHg = np.dot(g.T, Hg) # scalar

        v = pow(gHg, 0.5) * (x / xg - Hg / gHg)

        u = phi * (np.dot(grad, x) / np.dot(grad, Hg)) + (1 - phi) * xg / gHg

        HggH = np.dot(H, np.dot(np.outer(g, g), H))
        H = u * (H - HggH / gHg + theta * np.outer(v, v)) + xx / xg

        direction = -np.dot(H, grad)
        return direction, {'gradient_diff': g}

    def __iter__(self):
        args, kwargs = next(self.args)
        # gradient (first derivative)
        grad = self.fprime(self.wrt, *args, **kwargs)
        # previous gradient
        grad_m1 = scipy.zeros(grad.shape)

        if self.inv_hessian is None:
            # I
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:
                direction, info = -grad, {}
            else:
                direction, info = self.find_direction(
                    grad_m1, grad, step, self.inv_hessian)

            if not is_nonzerofinite(direction):
                # TODO: inform the user here.
                break

            step_length = self.line_search.search(
                direction, None, args, kwargs)

            # Update steps
            if step_length != 0:
                step = step_length * direction
                self.wrt += step
            else:
                self.logfunc(
                    {'message': 'step length is 0--need to bail out.'})
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad

            info.update({
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
            })
            yield info

class Bfgs(Minimizer):
    """BFGS (Broyden-Fletcher-Goldfarb-Shanno) is one of the most well-knwon
    quasi-Newton methods. The main idea is to iteratively construct an approximate inverse
    Hessian :math:`B^{-1}_t` by a rank-2 update:

        .. math::
            B^{-1}_{t+1} = B^{-1}_t + (1 + \\frac{y_t^TB^{-1}_ty_t}{y_t^Ts_t})\\frac{s_ts_t^T}{s_t^Ty_t} - \\frac{s_ty_t^TB^{-1}_t + B^{-1}_ty_ts_t^T}{s_t^Ty_t},

    where :math:`y_t = f(\\theta_{t+1}) - f(\\theta_{t})` and :math:`s_t = \\theta_{t+1} - \\theta_t`.

    The storage requirements for BFGS scale quadratically with the number of
    variables. For detailed derivations, see [nocedal2006a]_, chapter 6.

    .. [nocedal2006a]  Nocedal, J. and Wright, S. (2006),
        Numerical Optimization, 2nd edition, Springer.

    Attributes
    ----------
    wrt : array_like
        Current solution to the problem. Can be given as a first argument to \
        ``.f`` and ``.fprime``.

    f : Callable
        The object function.

    fprime : Callable
        First derivative of the objective function. Returns an array of the \
        same shape as ``.wrt``.

    initial_inv_hessian : array_like
        The initial estimate of the approximiate Hessian.

    line_search : LineSearch object.
        Line search object to perform line searches with.

    args : iterable
        Iterator over arguments which ``fprime`` will be called with.

    """

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None):
        """Create a BFGS object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``f`` and ``fprime`` should accept this array as a first argument.

        f : callable
            The objective function.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        initial_inv_hessian : array_like
            The initial estimate of the approximiate Hessian.

        line_search : LineSearch object.
            Line search object to perform line searches with.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        super(Bfgs, self).__init__(wrt, args=args)
        self.f = f
        self.fprime = fprime
        self.inv_hessian = initial_inv_hessian

        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')

    def find_direction(self, grad_m1, grad, step, inv_hessian):
        H = self.inv_hessian
        grad_diff = grad - grad_m1

        # scalar
        ys = np.inner(grad_diff, step)
        # grad-shape
        Hy = np.dot(H, grad_diff)
        # scalar
        yHy = np.inner(grad_diff, Hy)
        H += (ys + yHy) * np.outer(step, step) / ys ** 2
        H -= (np.outer(Hy, step) + np.outer(step, Hy)) / ys
        direction = -np.dot(H, grad)
        return direction, {'gradient_diff': grad_diff}

    def __iter__(self):
        args, kwargs = next(self.args)
        # gradient (first derivative)
        grad = self.fprime(self.wrt, *args, **kwargs)
        # previous gradient
        grad_m1 = scipy.zeros(grad.shape)

        if self.inv_hessian is None:
            # I
            self.inv_hessian = scipy.eye(grad.shape[0])

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i == 0:
                direction, info = -grad, {}
            else:
                direction, info = self.find_direction(
                    grad_m1, grad, step, self.inv_hessian)

            if not is_nonzerofinite(direction):
                # TODO: inform the user here.
                break

            step_length = self.line_search.search(
                direction, None, args, kwargs)

            # Update steps
            if step_length != 0:
                step = step_length * direction
                self.wrt += step
            else:
                self.logfunc(
                    {'message': 'step length is 0--need to bail out.'})
                break

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad

            info.update({
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
            })
            yield info

class Sbfgs(Bfgs):
    # TODO document

    def __init__(self, wrt, f, fprime, initial_inv_hessian=None,
                 line_search=None, args=None):
        # TODO document
        super(Sbfgs, self).__init__(
            wrt, f, fprime, line_search, args=args)

    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')


    def find_direction(self, grad_m1, grad, step, inv_hessian):
        # TODO document
        H = inv_hessian
        grad_diff = grad - grad_m1
        ys = np.inner(grad_diff, step)
        Hy = np.dot(H, grad_diff)
        yHy = np.inner(grad_diff, Hy)
        gamma = ys / yHy
        v = scipy.sqrt(yHy) * (step / ys - Hy / yHy)
        v = scipy.real(v)
        H[:] = gamma * (H - np.outer(Hy, Hy) / yHy + np.outer(v, v))
        H += np.outer(step, step) / ys
        direction = -np.dot(H, grad)
        return direction, {}

class Lbfgs(Minimizer):
    """l-BFGS (limited-memory BFGS) is a limited memory variation of the well-known
    BFGS algorithm. The storage requirement for BFGS scale quadratically with the number of variables,
    and thus it tends to be used only for smaller problems. Limited-memory BFGS reduces the
    storage by only using the :math:`l` latest updates (factors) in computing the approximate Hessian inverse
    and representing this approximation only implicitly. More specifically, it stores the last
    :math:`l` BFGS update vectors :math:`y_t` and :math:`s_t` and uses these to implicitly perform
    the matrix operations of BFGS (see [nocedal2006a]_).

    .. note::
       In order to handle simple box constraints, consider ``scipy.optimize.fmin_l_bfgs_b``.

    Attributes
    ----------
    wrt : array_like
        Current solution to the problem. Can be given as a first argument to \
        ``.f`` and ``.fprime``.

    f : Callable
        The object function.

    fprime : Callable
        First derivative of the objective function. Returns an array of the \
        same shape as ``.wrt``.

    initial_hessian_diag : array_like
        The initial estimate of the diagonal of the Hessian.

    n_factors : int
        The number of factors that should be used to implicitly represent the inverse Hessian.

    line_search : LineSearch object.
        Line search object to perform line searches with.

    args : iterable
        Iterator over arguments which ``fprime`` will be called with.

    """

    def __init__(self, wrt, f, fprime, initial_hessian_diag=1,
                 n_factors=10, line_search=None,
                 args=None):
        """
        Create an Lbfgs object.

        Attributes
        ----------
        wrt : array_like
            Current solution to the problem. Can be given as a first argument to \
            ``.f`` and ``.fprime``.

        f : Callable
            The object function.

        fprime : Callable
            First derivative of the objective function. Returns an array of the \
            same shape as ``.wrt``.

        initial_hessian_diag : array_like
            The initial estimate of the diagonal of the Hessian.

        n_factors : int
            The number of factors that should be used to implicitly represent the inverse Hessian.

        line_search : LineSearch object.
            Line search object to perform line searches with.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.

        """

        super(Lbfgs, self).__init__(wrt, args=args)

        self.f = f
        self.fprime = fprime
        self.initial_hessian_diag = initial_hessian_diag
        self.n_factors = n_factors
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(wrt, self.f, self.fprime)

    def set_from_info(self, info):
        raise NotImplemented('nobody has found the time to implement this yet')

    def extended_info(self, **kwargs):
        raise NotImplemented('nobody has found the time to implement this yet')

    def find_direction(self, grad_diffs, steps, grad, hessian_diag, idxs):
        grad = grad.copy()  # We will change this.
        n_current_factors = len(idxs)

        # TODO: find a good name for this variable.
        rho = scipy.empty(n_current_factors)
        alpha = scipy.empty(n_current_factors)

        for i in idxs[::-1]:
            rho[i] = 1 / scipy.inner(grad_diffs[i], steps[i])
            alpha[i] = rho[i] * scipy.inner(steps[i], grad)
            grad -= alpha[i] * grad_diffs[i]
        z = hessian_diag * grad

        # TODO: find a good name for this variable (surprise!)

        for i in idxs:
            beta = rho[i] * scipy.inner(grad_diffs[i], z)
            z += steps[i] * (alpha[i] - beta)

        return z, {}

    def __iter__(self):
        args, kwargs = next(self.args)
        grad = self.fprime(self.wrt, *args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)
        factor_shape = self.n_factors, self.wrt.shape[0]
        grad_diffs = scipy.zeros(factor_shape)
        steps = scipy.zeros(factor_shape)
        hessian_diag = self.initial_hessian_diag
        step_length = None
        step = scipy.empty(grad.shape)
        grad_diff = scipy.empty(grad.shape)

        # We need to keep track in which order the different statistics
        # from different runs are saved.
        #
        # Why?
        #
        # Each iteration, we save statistics such as the difference between
        # gradients and the actual steps taken. These are then later combined
        # into an approximation of the Hessian. We call them factors. Since we
        # don't want to create a new matrix of factors each iteration, we
        # instead keep track externally, which row of the matrix corresponds
        # to which iteration. `idxs` now is a list which maps its i'th element
        # to the corresponding index for the array. Thus, idx[i] contains the
        # rowindex of the for the (n_factors - i)'th iteration prior to the
        # current one.
        idxs = []
        for i, (next_args, next_kwargs) in enumerate(self.args):

            # sTgd = scipy.inner(step, grad_diff)
            # if sTgd > 1E-10:
            #     # Don't do an update if this value is too small.
            #     # Determine index for the current update.
            #     if not idxs:
            #         # First iteration.
            #         this_idx = 0
            #     elif len(idxs) < self.n_factors:
            #         # We are not "full" yet. Thus, append the next idxs.
            #         this_idx = idxs[-1] + 1
            #     else:
            #         # we are full and discard the first index.
            #         this_idx = idxs.pop(0)

            #     idxs.append(this_idx)
            #     grad_diffs[this_idx] = grad_diff
            #     steps[this_idx] = step

            # direction = lbfgs_two_loop_recursion(grad.copy(), steps, grad_diffs, idxs)
            # info = {}

            if i == 0:
                direction = -grad
                info = {}
            else:
                sTgd = scipy.inner(step, grad_diff)
                if sTgd > 1E-10:
                    # Don't do an update if this value is too small.
                    # Determine index for the current update.
                    if not idxs:
                        # First iteration.
                        this_idx = 0
                    elif len(idxs) < self.n_factors:
                        # We are not "full" yet. Thus, append the next idxs.
                        this_idx = idxs[-1] + 1
                    else:
                        # we are full and discard the first index.
                        this_idx = idxs.pop(0)

                    idxs.append(this_idx)
                    grad_diffs[this_idx] = grad_diff
                    steps[this_idx] = step
                    hessian_diag = sTgd / scipy.inner(grad_diff, grad_diff)

                direction, info = self.find_direction(
                    grad_diffs, steps, -grad, hessian_diag, idxs)

            if not is_nonzerofinite(direction):
                warnings.warn('search direction is either 0, nan or inf')
                break

            step_length = self.line_search.search(
                direction, None, args, kwargs)

            step[:] = step_length * direction
            if step_length != 0:
                self.wrt += step
            else:
                warnings.warn('step length is 0')
                pass

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            # TODO: not all line searches have .grad!
            grad_m1[:], grad[:] = grad, self.line_search.grad
            grad_diff = grad - grad_m1

            info.update({
                'step_length': step_length,
                'n_iter': i,
                'args': args,
                'kwargs': kwargs,
                'loss': self.line_search.val,
                'gradient': grad,
                'gradient_m1': grad_m1,
            })
            yield info
