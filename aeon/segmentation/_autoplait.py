from copy import deepcopy
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from aeon.segmentation.base import BaseSegmenter

ZERO = 1.e-10
N_INFER_ITER_HMM = 1
SEGMENT_R = 1.e-2
REGIME_R = 3.e-2
MAXBAUMN = 100
FB = 4 * 8
LM = .1
RM = True

class AutoPlaitSegmenter(BaseSegmenter):
    """AutoPlait Segmentation.

        Using ClaSP [1]_, [2]_ for the CPD problem is straightforward: We first compute the
        profile and then choose its global maximum as the change point. The following CPDs
        are obtained using a bespoke recursive split segmentation algorithm.

        Parameters
        ----------
        parallel : bool, default = True
            Param desc.
        min_k : int, default = 1
            Param desc.
        max_k : int, default = 8
            Param desc.
        max_seg : int, default = 100
            Param desc.
        n_sample : int, default = 10
            Param desc.
        infer_iter_min : int, default = 2
            Param desc.
        infer_iter_max : int, default = 10
            Param desc.
        random_seed : int, default = 41
            Seed to use for HMM initialisation.

        References
        ----------
        .. [1] AUTOPLAIT REFERENCE.
        .. [2] Schafer, Patrick and Ermshaus, Arik and Leser, Ulf. "ClaSP - Time Series
        Segmentation", CIKM, 2021.

        Examples
        --------
        >> from aeon.segmentation import AutoPlaitSegmenter
        >> from aeon.datasets import load_gun_point_segmentation
        >> X, _, cps = load_gun_point_segmentation()
        >> X = np.array(X)
        >> autoplait = AutoPlaitSegmenter()
        >> found_cps = autoplait.fit_predict(X)
        """

    _tags = {
        "returns_dense": True,
        "fit_is_empty": True,
        "capability:multivariate": True,
        "python_dependencies": "hmmlearn"
    }

    def __init__(self,
                 parallel = True,
                 min_k = 1,
                 max_k = 8,
                 max_seg = 100,
                 n_sample = 10,
                 infer_iter_min = 2,
                 infer_iter_max = 10,
                 random_seed=41):
        self.parallel = parallel
        self.min_k = min_k
        self.max_k = max_k
        self.max_seg = max_seg
        self.n_sample = n_sample
        self.infer_iter_min = infer_iter_min
        self.infer_iter_max = infer_iter_max
        self.random_seed = random_seed

        self.costT = np.inf
        self.regimes = []
        super().__init__(axis=0)

    def _predict(self, X) -> np.ndarray:
        """Perform AutoPlait segmentation.

        Parameters
        ----------
        X: np.ndarray
            2D time series shape (n_timepoints, n_channels).

        Returns
        -------
        y_pred : np.ndarray
            Predicted change points of the time series
        """
        self.X = X
        self.n, self.d = self.X.shape
        reg = _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed)
        reg.add_segment(0, self.n)
        reg.estimate_hmm(self.X)
        candidates = [reg]

        while candidates:
            self.costT = _mdl_total(self.regimes, candidates)
            reg = candidates.pop()

            # try to split regime: s0, s1
            reg0, reg1 = self._regime_split(X, reg)
            # print(reg0.subs[:reg0.n_seg], '0')
            # print(reg1.subs[:reg1.n_seg], '1')
            costT_s01 = reg0.costT + reg1.costT + REGIME_R * reg.costT
            print(f'\t-- try to split: {costT_s01:.6} vs {reg.costT:.6}')
            # print(s0.costT, s1.costT)

            if costT_s01 < reg.costT:
                candidates.append(reg0)
                candidates.append(reg1)
            else:
                self.regimes.append(reg)

        m = 0
        for i in range(len(self.regimes)):
            m += self.regimes[i].n_seg
        self.m = m

        return self._regimes_to_change_points()

    def complete_parameters(self):
        return {'m': self.m,
                'n': self.n,
                'd': self.d}

    def _regimes_to_change_points(self):
        """
        Convert the list [_Regime] into a list of change points
        @return: A list of change points
        """
        cps = []
        for i in range(len(self.regimes)):
            for j in range(len(self.regimes[i].subs[:self.regimes[i].n_seg])):
                cps.append(self.regimes[i].subs[:self.regimes[i].n_seg][j][0])
        if cps == [0]: cps = [] # The only segment start is 0, means no segments (the entire TS is a single segment)
        return np.array(sorted(cps))


    def _regime_split(self, X, sx):
        opt0, opt1 = (_Regime(self.max_seg, self.min_k, self.max_k, self.random_seed),
                      _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed))
        n, d = X.shape
        seedlen = int(n * LM)
        s0, s1 = self._find_centroid(X, sx, self.n_sample, seedlen)
        if not s0.n_seg or not s1.n_seg:
            return opt0, opt1
        for i in range(self.infer_iter_max):
            select_largest(s0)
            select_largest(s1)
            s0.estimate_hmm(X)
            s1.estimate_hmm(X)
            self._cut_point_search(X, sx, s0, s1, remove_noise=RM)
            if not s0.n_seg or not s1.n_seg:
                break
            diff = (opt0.costT + opt1.costT) - (s0.costT + s1.costT)
            if diff > 0:
                copy_segments(s0, opt0)
                copy_segments(s1, opt1)
            elif i >= self.infer_iter_min:
                break
        copy_segments(opt0, s0)
        copy_segments(opt1, s1)
        del opt0, opt1
        if not s0.n_seg or not s1.n_seg:
            return s0, s1
        s0.estimate_hmm(X)
        s1.estimate_hmm(X)
        return s0, s1

    def _cut_point_search(self, X, sx, s0, s1, remove_noise=True):
        s0.initialize()
        s1.initialize()
        lh = 0.
        for i in range(sx.n_seg):
            lh += _search_aux(X, sx.subs[i, 0], sx.subs[i, 1], s0, s1)
        if remove_noise: self._remove_noise(X, sx, s0, s1)
        s0.compute_lh_mdl(X)
        s1.compute_lh_mdl(X)
        return lh

    def _find_centroid_wrap(self, X, Sx, seedlen, idx0, idx1, u):
        s0, s1 = self._uniform_sampling(seedlen, idx0, idx1, u)
        if not s0.n_seg or not s1.n_seg:
            return np.inf, None, None
        subs0 = s0.subs[0]
        subs1 = s1.subs[0]
        s0.estimate_hmm_k(X, self.min_k)
        s1.estimate_hmm_k(X, self.min_k)
        self._cut_point_search(X, Sx, s0, s1, False)
        if not s0.n_seg or not s1.n_seg:
            return np.inf, None, None
        costT_s01 = s0.costT + s1.costT
        return costT_s01, subs0, subs1

    def _find_centroid(self, X, Sx, n_samples, seedlen):
        u = self._uniformset(Sx, n_samples, seedlen)
        # print(u.subs[:u.n_seg], u.n_seg)

        if self.parallel:
            results = Parallel(n_jobs=4)(
                [delayed(self._find_centroid_wrap)(X, Sx, seedlen, iter1, iter2, u)
                 for iter1, iter2 in combinations(range(u.n_seg), 2)])
        else:
            results = []
            for iter1, iter2 in combinations(range(u.n_seg), 2):
                results.append(self._find_centroid_wrap(X, Sx, seedlen, iter1, iter2, u))

        # pp.pprint(results)
        if not results:
            print('fixed sampling')
            s0, s1 = self._fixed_sampling(Sx, seedlen)
            return s0, s1
        centroid = np.argmin([res[0] for res in results])
        # print(results[centroid])
        costMin, seg0, seg1 = results[centroid]
        if costMin == np.inf:
            print('!! --- centroid not found')
            # s0, s1 = fixed_sampling(X, Sx, seedlen)
            # print('fixed_sampling', s0.subs, s1.subs)
            return (_Regime(self.max_seg, self.min_k, self.max_k, self.random_seed),
                    _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed))
        s0, s1 = (_Regime(self.max_seg, self.min_k, self.max_k, self.random_seed),
                  _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed))
        s0.add_segment(seg0[0], seg0[1])
        s1.add_segment(seg1[0], seg1[1])
        # print(s0.n_seg, s1.n_seg)
        # time.sleep(3)
        return s0, s1

    def _scan_mindiff(self, X, Sx, s0, s1):
        loc0, _ = find_mindiff(X, s0, s1)
        loc1, _ = find_mindiff(X, s1, s0)
        # print(s0.subs[loc0], s1.subs[loc1])
        if (loc0 == -1 or loc1 == -1
                or s0.subs[loc0, 1] < 2
                or s1.subs[loc1, 1] < 2):
            return np.inf
        tmp0 = _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed)
        tmp1 = _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed)
        st, ln = s0.subs[loc0]
        tmp0.add_segment(st, ln)
        st, ln = s1.subs[loc1]
        tmp1.add_segment(st, ln)
        tmp0.estimate_hmm_k(X, self.min_k)
        tmp1.estimate_hmm_k(X, self.min_k)
        costC = self._cut_point_search(X, Sx, tmp0, tmp1, False)
        del tmp0, tmp1
        return costC

    def _remove_noise(self, X, Sx, s0, s1):
        if s0.n_seg <= 1 and s1.n_seg <= 1:
            return
        per = SEGMENT_R
        remove_noise_aux(X, Sx, s0, s1, per)
        costC = self._scan_mindiff(X, Sx, s0, s1)
        opt0 = _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed)
        opt1 = _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed)
        copy_segments(s0, opt0)
        copy_segments(s1, opt1)
        prev = np.inf
        while per <= SEGMENT_R * 10:
            if costC >= np.inf:
                break
            per *= 2
            remove_noise_aux(X, Sx, s0, s1, per)
            if s0.n_seg <= 1 or s1.n_seg <= 1:
                break
            costC = self._scan_mindiff(X, Sx, s0, s1)
            if prev > costC:
                copy_segments(s0, opt0)
                copy_segments(s1, opt1)
                prev = costC
            else:
                break
        copy_segments(opt0, s0)
        copy_segments(opt1, s1)
        # _estimate_hmm(X, s0)
        # _estimate_hmm(X, s1)
        del opt0, opt1

    def _uniformset(self, Sx, n_samples, seedlen):
        u = _Regime(self.max_seg, self.min_k, self.max_k, self.random_seed)
        w = int((Sx.len - seedlen) / n_samples)
        for i in range(Sx.n_seg):
            if u.n_seg >= n_samples:
                return u
            st, ln = Sx.subs[i]
            ed = st + ln
            for j in range(n_samples):
                nxt = st + j * w
                if nxt + seedlen > ed:
                    st = ed - seedlen
                    if st < 0: st = 0
                    u.add_segment_ex(st, seedlen)
                    break
                u.add_segment_ex(nxt, seedlen)
        return u

    def _fixed_sampling(self, Sx, seedlen):
        # print('nseg', Sx.n_seg)
        s0, s1 = (_Regime(self.max_seg,
                         self.min_k,
                         self.max_k,
                         self.random_seed),
                  _Regime(self.max_seg,
                          self.min_k,
                          self.max_k,
                          self.random_seed))
        loc = 0 % Sx.n_seg
        r = Sx.subs[loc, 0]
        if Sx.n_seg == 1:
            dt = Sx.subs[0, 1]
            if dt < seedlen:
                s0.add_segment(r, dt)
                s1.add_segment(r, dt)
            else:
                s0.add_segment(r, dt)
                s1.add_segment(r, dt)
        s0.add_segment(r, seedlen)
        loc = 1 % Sx.n_seg
        r = Sx.subs[loc, 0] + int(Sx.subs[loc, 1] / 2)
        s1.add_segment(r, seedlen)
        return s0, s1

    def _uniform_sampling(self, length, n1, n2, u):
        s0, s1 = (_Regime(self.max_seg,
                         self.min_k,
                         self.max_k,
                         self.random_seed),
                  _Regime(self.max_seg,
                          self.min_k,
                          self.max_k,
                          self.random_seed))
        i, j = int(n1 % u.n_seg), int(n2 % u.n_seg)
        # print(i, j)
        st0, st1 = u.subs[i, 0], u.subs[j, 0]
        if abs(st0 - st1) < length:
            return s0, s1
        s0.add_segment(st0, length)
        s1.add_segment(st1, length)
        return s0, s1

class _Regime:
    def __init__(self,
                 max_seg,
                 min_k,
                 max_k,
                 random_seed):
        self.max_seg = max_seg
        self.min_k = min_k
        self.max_k = max_k
        self.random_seed = random_seed
        self.subs = np.zeros((self.max_seg, 2), dtype=np.int16)
        self.model = None
        self.delta = 1.
        self.initialize()

    def initialize(self):
        self.len = 0
        self.n_seg = 0
        self.costC = np.inf
        self.costT = np.inf

    def add_segment(self, st, dt):
        if dt <= 0: return
        st = 0 if st < 0 else st
        n_seg = self.n_seg
        if n_seg == self.max_seg:
            raise ValueError(" ")
        elif n_seg == 0:
            self.subs[0, :] = (st, dt)
            self.n_seg += 1
            self.len = dt
            self.delta = 1 / dt
        else:
            loc = 0
            while loc < n_seg:
                if st < self.subs[loc, 0]:
                    break
                loc += 1
            self.subs[loc+1:n_seg+1, :] = self.subs[loc:n_seg, :]
            self.subs[loc, :] = (st, dt)
            n_seg += 1
            # remove overlap
            curr = np.inf
            while curr > n_seg:
                curr = n_seg
                for i in range(curr - 1):
                    st0, dt0 = self.subs[i]
                    st1, dt1 = self.subs[i + 1]
                    ed0, ed1 = (st0 + dt0), (st1 + dt1)
                    ed = ed0 if ed0 > ed1 else ed1
                    if ed0 > st1:
                        self.subs[i+1:-1, :] = self.subs[i+2:, :]  # pop subs[i]
                        self.subs[i, 1] = ed - st0
                        n_seg -= 1
                        break
            self.n_seg = n_seg
            self.len = sum(self.subs[:n_seg, 1])
            self.delta = self.n_seg / self.len

    def add_segment_ex(self, st, dt):
        self.subs[self.n_seg, :] = (st, dt)
        self.n_seg += 1
        self.len += dt
        self.delta = self.n_seg / self.len

    def del_segment(self, loc):
        seg = self.subs[loc]
        self.subs[loc:-1, :] = self.subs[loc+1:, :]  # pop subs[i]
        self.n_seg -= 1
        self.len -= seg[1]
        self.delta = self.n_seg / self.len if self.len > 0 else ZERO
        return seg

    def estimate_hmm_k(self, X, k=1):
        from hmmlearn.hmm import GaussianHMM
        X_, lengths = self._parse_input(X)
        self.model = GaussianHMM(n_components=k,
                                   covariance_type='diag',
                                   n_iter=N_INFER_ITER_HMM,
                                   random_state=self.random_seed)
        self.model.fit(X_, lengths=lengths)
        self.delta = self.n_seg / self.len

    def _parse_input(self, X):
        n_seg = self.n_seg
        if n_seg == 1:
            st, dt = self.subs[0]
            return X[st:st + dt, :], [dt]
        n_seg = MAXBAUMN if n_seg > MAXBAUMN else n_seg
        subss = []
        lengths = []
        for st, dt in self.subs[:n_seg]:
            subss.append(X[st:st + dt, :])
            lengths.append(dt)
        subss = np.concatenate(subss)
        return subss, lengths

    def _mdl(self):
        m = self.n_seg
        k = self.model.n_components
        d = self.model.n_features
        costLen = 0.
        costC = self.costC
        costM = cost_hmm(k, d)
        for i in range(m):
            costLen += np.log2(self.subs[i, 1])
        costLen += m * np.log2(k)
        return costC + costM + costLen

    def estimate_hmm(self, X):
        self.costT = np.inf
        opt_k = self.min_k
        for k in range(self.min_k, self.max_k):
            prev = self.costT
            self.estimate_hmm_k(X, k)
            self.compute_lh_mdl(X)
            if self.costT > prev:
                opt_k = k - 1
                break
        if opt_k < self.min_k: opt_k = self.min_k
        if opt_k > self.max_k: opt_k = self.max_k
        self.estimate_hmm_k(X, opt_k)
        self.compute_lh_mdl(X)

    def compute_lh_mdl(self, X):
        if self.n_seg == 0:
            self.costT = self.costC = np.inf
            return
        self.costC = 0.
        for i in range(self.n_seg):
            st, dt = self.subs[i]
            self.costC += _viterbi(X[st:st + dt], self.model, self.delta)
        self.costT = self._mdl()
        if self.costT < 0:
            self.costT = np.inf  # avoid overfitting

def _mdl_total(stack0, stack1):
    r = len(stack0) + len(stack1)
    m = sum([regime.n_seg for regime in stack0])
    m += sum([regime.n_seg for regime in stack1])
    costT = mdl_segment(stack0) + mdl_segment(stack1)
    costT += log_s(r) + log_s(m) + m * np.log2(r) + FB * r ** 2
    # print(f'[r, m, total_cost] = {r}, {m}, {costT:.6}')
    print('====================')
    print(' r:\t', r)
    print(' m:\t', m)
    print(f' costT:\t{costT:.6}')
    print('====================')
    return costT

def _search_aux(X, st, dt, s0, s1):
    d0, d1 = s0.delta, s1.delta
    if d0 <= 0 or d1 <= 0: raise ValueError('delta is zero.')
    m0, m1 = s0.model, s1.model
    k0, k1 = m0.n_components, m1.n_components
    Pu, Pv = np.zeros(k0), np.zeros(k0)  # log probability
    Pi, Pj = np.zeros(k1), np.zeros(k1)  # log probability
    Su, Sv = [[] for _ in range(k0)], [[] for _ in range(k0)]
    Si, Sj = [[] for _ in range(k1)], [[] for _ in range(k1)]

    t = st
    Pv = np.log(d1) + np.log(m0.startprob_ + ZERO)
    for v in range(k0):
        Pv[v] += gaussian_pdfl(X[t], m0.means_[v], m0.covars_[v])
    Pj = np.log(d0) + np.log(m1.startprob_ + ZERO)
    for j in range(k1):
        Pj[j] += gaussian_pdfl(X[t], m1.means_[j], m1.covars_[j])

    for t in range(st + 1, st + dt):
        # Pu(t)
        maxj = np.argmax(Pj)
        for u in range(k0):
            maxPj = Pj[maxj] + np.log(d1) + np.log(m0.startprob_[u] + ZERO) + gaussian_pdfl(X[t], m0.means_[u], m0.covars_[u])
            val = Pv + np.log(1. - d0) + np.log(m0.transmat_[:, u] + ZERO)
            for v in range(k0):
                val[v] += gaussian_pdfl(X[t], m0.means_[u], m0.covars_[u])
            maxPv, maxv = np.max(val), np.argmax(val)
            if maxPj > maxPv:
                Pu[u] = maxPj
                Su[u] = deepcopy(Sj[maxj])
                Su[u].append(t)
            else:
                Pu[u] = maxPv
                Su[u] = deepcopy(Sv[maxv])
        # Pj(t)
        maxv = np.argmax(Pv)
        for i in range(k1):
            maxPv = Pv[maxv] + np.log(d0) + np.log(m1.startprob_[i] + ZERO) + gaussian_pdfl(X[t], m1.means_[i], m1.covars_[i])
            val = Pj + np.log(1. - d1) + np.log(m1.transmat_[:, i] + ZERO)
            for j in range(k1):
                val[j] += gaussian_pdfl(X[t], m1.means_[i], m1.covars_[i])
            maxPj, maxj = np.max(val), np.argmax(val)
            if maxPv > maxPj:
                Pi[i] = maxPv
                Si[i] = deepcopy(Sv[maxv])
                Si[i].append(t)
            else:
                Pi[i] = maxPj
                Si[i] = deepcopy(Sj[maxj])
        tmp = np.copy(Pu); Pu = np.copy(Pv); Pv = np.copy(tmp)
        tmp = np.copy(Pi); Pi = np.copy(Pj); Pj = np.copy(tmp)
        tmp = deepcopy(Su); Su = deepcopy(Sv); Sv = deepcopy(tmp)
        tmp = deepcopy(Si); Si = deepcopy(Sj); Sj = deepcopy(tmp)

    maxv = np.argmax(Pv)
    maxj = np.argmax(Pj)
    if Pv[maxv] > Pj[maxj]:
        path = Sv[maxv]
        firstID = pow(-1, len(path)) * 1
        llh = Pv[maxv]
    else:
        path = Sj[maxj]
        firstID = pow(-1, len(path)) * -1
        llh = Pj[maxj]

    curst = st
    for i in range(len(path)):
        nxtst = path[i]
        if firstID * pow(-1, i) == 1:
            s0.add_segment(curst, nxtst - curst)
        else:
            s1.add_segment(curst, nxtst - curst)
        curst = nxtst
    if firstID * pow(-1, len(path)) == 1:
        s0.add_segment(curst, st + dt - curst)
    else:
        s1.add_segment(curst, st + dt - curst)
    # print(path)
    # print('s0', s0.subs[:s0.n_seg])
    # print('s1', s1.subs[:s1.n_seg])
    return -llh / np.log(2.)  # data coding cost

def _viterbi(X, hmm, delta):
    if not 0 <= delta <= 1:
        exit('not appropriate delta')
    # print(hmm.startprob_)
    llh = hmm.score(X) + np.log(delta) + np.log(1 - delta)
    return -llh / np.log(2)  # data coding cost

def log_s(x):
    return 2. * np.log2(x) + 1.

def cost_hmm(k, d):
    return FB * (k + k ** 2 + 2 * k * d) + 2. * np.log(k) / np.log(2.) + 1.

def mdl_segment(stack):
    return np.sum([regime.costT for regime in stack])

def gaussian_pdfl(x, means, covars):
    n_dim = len(x)
    covars = np.diag(covars)
    lpr = -.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars))
                  + np.sum((means ** 2) / covars)
                  - 2 * np.dot(x, (means / covars).T)
                  + np.dot(x ** 2, (1. / covars).T))
    return lpr

def find_mindiff(X, s0, s1):
    cost = np.inf
    loc = -1
    for i in range(s0.n_seg):
        st, dt = s0.subs[i]
        costC0 = _viterbi(X[st:st+dt], s0.model, s0.delta)
        costC1 = _viterbi(X[st:st+dt], s1.model, s1.delta)
        diff = abs(costC1 - costC0)
        if cost > diff:
            loc, cost = i, diff
    return loc, cost

def remove_noise_aux(X, Sx, s0, s1, per):
    if per == 0: return
    th = per * Sx.costT
    mprev = np.inf
    while mprev > s0.n_seg + s1.n_seg:
        mprev = s0.n_seg + s1.n_seg
        loc0, diff0 = find_mindiff(X, s0, s1)
        loc1, diff1 = find_mindiff(X, s1, s0)
        cost, idx = (diff0, 0) if diff0 < diff1 else (diff1, 1)
        if cost >= th:
            continue
        if idx == 0:
            st, dt = s0.del_segment(loc0)
            s1.add_segment(st, dt)
        else:
            st, dt = s1.del_segment(loc1)
            s0.add_segment(st, dt)

def copy_segments(s0, s1):  # from s0 to s1
    s1.subs = deepcopy(s0.subs)
    s1.n_seg = s0.n_seg
    s1.len = s0.len
    s1.costT = s0.costT
    s1.costC = s1.costC
    s1.delta = s0.delta

def select_largest(s):
    loc = np.argmax(s.subs[:, 1])
    st, dt = s.subs[loc]
    s.initialize()
    s.add_segment(st, dt)