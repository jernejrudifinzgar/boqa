from scipy.interpolate import interp1d
from bayes_opt import BayesianOptimization
from json import dumps
import qutip
from qutip import Qobj, qzero, sesolve, basis, tensor, identity
import numpy as np
import matplotlib.pyplot as plt




class AQC:
    """This will handle the evaluation of a certain adiabatic schedule."""

    def __init__(self, hamiltonians: list, total_time=10., integration_steps=None, fom='overlap', psi0=None,
                 percentile=1., nshots=500):
        # check if all hamiltonians have the same dimensions.
        assert all([h.dims == hamiltonians[0].dims for h in hamiltonians])
        # set hamiltonians and total time of adiabatic evolution
        self.hams = hamiltonians
        self.total_time = total_time
        # set the step density parameters
        if integration_steps is None:
            self.integration_steps = int(total_time * 10) + 1
        else:
            self.integration_steps = integration_steps
        # figure of merit
        self.fom = fom
        self.psi0 = psi0

        self.percentile = percentile
        self.nshots = nshots

    @property
    def _hamiltonian_dims(self):
        """QuTip operator dimensions."""
        return self.hams[0].dims[0]

    @property
    def nr_hamiltonians(self):
        return len(self.hams)

    def total_hamiltonian(self, t, path_list, parameters=None):
        """Total Hamiltonian QuTip operator."""
        if isinstance(path_list[0], np.ndarray):
            def interpolate(path_arr):
                return lambda tau, _: interp1d(np.linspace(0, self.total_time, self.integration_steps), path_arr)(tau)

            path_list = [interpolate(pl) for pl in path_list]

        if parameters is None:
            parameters = {"T": self.total_time}
        return qutip.Qobj(sum([ham * float(path(t, parameters)) for path, ham in zip(path_list, self.hams)]))

    def _qutip_ham(self, path_list, parameters):
        """Return the qutip Hamiltonian"""
        ctrls = []
        if not isinstance(path_list[0], np.ndarray):
            for path in path_list:
                ctrls.append(
                    np.array([path(t, parameters) for t in np.linspace(0, self.total_time, self.integration_steps)])
                )
        else:
            ctrls = path_list
        return [qzero(self._hamiltonian_dims)] + [[ham, ctrl] for ctrl, ham in zip(ctrls, self.hams)]

    def gap(self, t, path_list, parameters):
        """Spectral gap of the total Hamiltonian at the specified time t."""
        parameters['T'] = self.total_time
        ham = self.total_hamiltonian(t, path_list, parameters)
        return np.diff(ham.eigenenergies())[0]

    def run_protocol(self, path_list, parameters):
        """Running a protocol defined by paths and the parameters."""
        assert len(path_list) == len(self.hams)

        parameters['T'] = self.total_time
        # Hamiltonian in qutip format
        qutip_ham = self._qutip_ham(path_list, parameters)
        if self.psi0 is None:
            # we start in the ground state of the initial Hamiltonian
            psi0 = ground_state_opt(self.total_hamiltonian(0, path_list, parameters))
        else:
            psi0 = self.psi0
        # integration times
        times = np.linspace(0, self.total_time, int(self.integration_steps))

        # integrate the ODE
        final_state = sesolve(qutip_ham, psi0, times, args=parameters).states[-1]
        # final ham:
        final_ham = self.total_hamiltonian(self.total_time, path_list, parameters).tidyup()
        if isinstance(self.fom, Qobj):
            if is_diagonal(self.fom):
                energies = np.diag(self.fom).real
                probs = np.abs(np.array(final_state).ravel()) ** 2
            else:
                energies, eigenstates = self.fom.eigenstates()
                probs = [np.abs(final_state.overlap(es)) ** 2 for es in eigenstates]

            measured_vals = np.random.choice(energies, size=self.nshots, p=probs)
            accepted_vals = np.sort(measured_vals)[-(int(self.percentile * self.nshots)):]

            return np.mean(accepted_vals)

        elif self.fom == 'overlap':
            # we return the overlap between the ground state and the target state
            gs = ground_state_opt(final_ham)
            return np.abs(final_state.overlap(gs))
        elif self.fom == 'approx':
            # if the chosen figure of merit is the energy, we return the approximation ratio:
            if is_diagonal(final_ham):
                diagonal = np.diag(final_ham).real
                emin, emax = np.min(diagonal), np.max(diagonal)
            else:
                energies = final_ham.eigenenergies()
                emax, emin = np.max(energies), np.min(energies)
                # add sampling noise and return the approximation ratio
            return (emax - qutip.expect(final_ham, final_state)) / (emax - emin)
        elif self.fom == 'measurements':
            if is_diagonal(final_ham):
                energies = np.diag(final_ham).real
                probs = np.abs(np.array(final_state).ravel()) ** 2
            else:
                # plt.matshow(np.)
                energies, eigenstates = final_ham.eigenstates()
                probs = [np.abs(final_state.overlap(es)) ** 2 for es in eigenstates]
            measured_energies = np.random.choice(energies, size=self.nshots, p=probs)
            accepted_energies = np.sort(measured_energies)[:(int(self.percentile * self.nshots))]
            emax, emin = np.max(energies), np.min(energies)

            return (emax - np.mean(accepted_energies)) / (emax - emin)
# return qutip.expect(final_ham, final_state)
        elif self.fom == 'energy':
            return qutip.expect(final_ham, final_state)
        else:
            raise NotImplementedError(f'Figure of merit \"{self.fom}\" not implemented.')

    def gs_population(self, path_list, parameters, n_points=10, verbose=False):
        """This function tracks the ground state population along the adiabatic evolution. Returns the populations at
         timesteps given by times."""

        parameters['T'] = self.total_time

        psi0 = self.total_hamiltonian(0, path_list, parameters).groundstate()[1]

        # number of steps the integrator takes
        n_steps = int(self.integration_steps)
        # integration times
        times = np.linspace(0, self.total_time, n_steps)

        qutip_ham = self._qutip_ham(path_list, parameters)

        states = sesolve(qutip_ham, psi0, times, args=parameters).states
        # this determines the indices at which the ground state population is to be probed. It's this complicated as I
        # want to include the last time in every case.
        inds = sorted(list(set(list(range(0, n_steps, np.ceil(n_steps / n_points).astype(int))) + [n_steps - 1])))
        gs_populations = []
        ts_out = [times[ind] for ind in inds]
        for ind, t in zip(inds, ts_out):
            gs = self.total_hamiltonian(t, path_list, parameters).groundstate()[1]
            gs_populations.append(np.abs(gs.overlap(states[ind])))
            if verbose:
                print(f"Time {t:.2f} done.")

        return ts_out, gs_populations

    def visualize_controls(self, path_list, parameters):
        """A function to return the matplotlib plotting objects for a visualization of the adiabatic paths specified
        by paths and parameters."""
        parameters['T'] = self.total_time

        fig, ax = plt.subplots(figsize=(4, 3))
        times = np.linspace(0, self.total_time, num=int(self.integration_steps))
        for i, path in enumerate(path_list):
            ax.plot(times, [path(t, parameters) for t in times], label=f'path$_{i}$')
        ax.legend()
        return fig, ax


def is_diagonal(op):
    """Checks if an operator is diagonal."""
    return np.all(np.array(op) == np.diag(np.diag(op)))


def ground_state_opt(hamiltonian):
    """Finds the grounds state of a hamiltonian, but checks if it's diagonal first, such that unnecessary
    computations are avoided."""
    if is_diagonal(hamiltonian):
        state = basis(hamiltonian.shape[0], np.argmin(np.diag(hamiltonian)))
        ds = hamiltonian.dims[0]
        state.dims = [ds, [1] * len(ds)]
        return state
    else:
        return hamiltonian.groundstate()[1]


def fourier_path(nr_frequencies: int, which: int) -> callable:
    """Fourier path which takes in two ints.
    :param nr_frequencies: Number of frequencies used to specify the Fourier path.
    :param which: Specifies which parameters are to be used with this particular path. Since the parameters are
    specified in a dict with keys with name f"p{which}{i_frequency}". See also below in the coefs defenition.
    """
    # We add weights to the individual contributions, with higher order frequencies having less of an impact. This is to
    # ensure that the deviations from the linear ramp do not depend on how many parameters we choose to use for our path
    norm = sum(1 / (n + 1) ** 2 for n in range(nr_frequencies))

    def _path(t, args):
        """Function that is passed to the QuTip solver, and used as the parametrization."""
        ramp = t / args['T']
        coefs = [2 * (args[f'p{which}{i}'] - 0.5) / norm for i in range(nr_frequencies)]
        return ramp + sum(
            [coefs[i] * np.sin((i + 1) * np.pi * t / args['T']) for i in range(nr_frequencies)]
        )

    return _path


def bo_fourier(aqc_obj, path_list, param_nrs, ip=9, num_iter=50, decay_ratio=0.5, random_seed=42, return_target=True,
               acq='ucb', xi=0.05, nshots=500):
    """Use Bayesian optimization to optimize the parameters for Fourier-parametrized paths. Param_nrs specifies the
    number of parameters used for each path. ip is the number of initial guesses without any Bayesian decisions on what
    is the next best point to sample. n_iter is the number of iterations."""

    # We initialize the parameters such that they correspond to a linear path. Due to compatibility with real_space path
    # we choose 0.5 to correspond to a path which does not deviate from the linear ramp in that particular frequency.
    initial_parameters = [0.5] * sum(param_nrs)
    # We specify the bounds in which the Bayesian optimizer looks for solutions. These are bounded to ensure that
    # higher frequencies have a smaller contribution. Based on Physical intuitions about how the paths should look like.
    bounds = {f"p{wh}{nr}": (0.5 - 1 / (nr + 2) ** 2, 0.5 + 1 / (nr + 2) ** 2)
              for wh in range(len(param_nrs)) for nr in range(param_nrs[wh])}
    aqc_obj.nshots = nshots

    def _objective(**params):
        """Objective function which takes in the parameters and runs the protocol,
         returning the ground state overlap."""
        return aqc_obj.run_protocol(path_list, params)

    # optimizer instance
    optimizer = BayesianOptimization(f=_objective,
                                     pbounds=bounds,
                                     random_state=random_seed)
    optimizer.set_gp_params(alpha=np.sqrt(1 / (nshots * aqc_obj.percentile)))
    # specify initial point; linear schedule always serves as a baseline.
    optimizer.probe(initial_parameters, lazy=True)

    # this determines the point after which we start moving towards exploitation
    # (after decay_steps steps of optimization)
    decay_steps = int(num_iter * decay_ratio)
    # this determines the speed with which we decay the exploration rate.
    kappa_decay = 0.01 ** (1 / decay_steps)
    # this is the relevant parameter passed to the maximize function, telling when to start decaying exploration.
    kappa_decay_delay = int(num_iter - decay_steps)
    # optimize:
    optimizer.maximize(init_points=ip, n_iter=num_iter, acq=acq, xi=xi,
                       kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay)
    if return_target:
        # we return the final target value, i.e. the best overlap achieved during our optimization.
        return optimizer.max['target']
    else:
        # we return the optimizer object.
        return optimizer


def real_path(nr_points, which, kind='linear'):
    def _linear_path(t, args):
        if t >= args['T']:
            # taking care of boundary errors.
            return 1
        else:
            # manual linear interpolation
            points = np.array([0.] + [args[f'p{which}{i}'] for i in range(nr_points)] + [1.])
            wh = int((nr_points + 1) * t / args['T'])
            return points[wh] + (points[wh + 1] - points[wh]) * (t * (nr_points + 1) / args['T'] - wh)

    def _cubic_path(t, args):
        if t >= args['T']:
            return 1.0
        else:
            points = np.array([0.] + [args[f'p{which}{i}'] for i in range(nr_points)] + [1.])
            return interp1d(np.linspace(0, args['T'], len(points)), points, kind='cubic')(t)

    if kind == 'linear':
        return _linear_path
    elif kind == 'cubic':
        return _cubic_path
    else:
        raise NotImplementedError('kind not implemented.')
    

def sigz(j):
    """Sigma-z operator in the reduced Hilbert space of conserved total angular momentum."""
    return qutip.Qobj(np.diag(np.arange(-j, j + 1)))


def sigx(j):
    """Sigma-x operator in the reduced Hilbert space of conserved total angular momentum."""
    d = int(2 * j + 1)
    arr = np.zeros((d, d))

    for x, m in enumerate(np.arange(-j, j + 1)):
        if x + 1 < d:
            arr[x, x + 1] += np.sqrt(j * (j + 1) - m * (m + 1))
        if x - 1 >= 0:
            arr[x, x - 1] += np.sqrt(j * (j + 1) - m * (m - 1))

    return qutip.Qobj(arr) / 2


def pspin(n, c=0, p=3):
    """p-spin hamiltonian in the J=max subspace."""
    dim1, dim2 = round(n * c), n - round(n * c)
    return - n * ((2 / n) * (tensor(sigz(dim1 / 2), identity(dim2 + 1)) +
                             tensor(identity(dim1 + 1), sigz(dim2 / 2)))
                  ) ** p


def h_init(n, c):
    """Initial solution hamiltonian in the J=max subspace."""
    dim1, dim2 = round(n * c), n - round(n * c)
    return -2 * (tensor(sigz(dim1 / 2), identity(dim2 + 1)) - tensor(identity(dim1 + 1), sigz(dim2 / 2)))


def v_tf(n, c=0):
    """Transverse field hamiltonian in the J=max subspace"""
    dim1, dim2 = round(n * c), n - round(n * c)
    return -2 * (tensor(sigx(dim1 / 2), identity(dim2 + 1)) + tensor(identity(dim1 + 1), sigx(dim2 / 2)))



def linear_path():
    """A helper function that generates a linear path."""
    return fourier_path(0, 0)


def bo_realspace(aqc_obj, path_list, param_nrs, ip=9, num_iter=50, decay_ratio=0.5, span=1.,
                 random_seed=42, return_target=True, acq='ucb', xi=0.05, nshots=500, bounds_in=None):
    """Bayesian optimization for a path parametrized in real space."""
    if bounds_in is None:
        bounds = {f"p{wh}{nr}": (max((nr + 1) / (param_nrs[wh] + 1) - span / (param_nrs[wh] + 1), 0),
                                 min((nr + 1) / (param_nrs[wh] + 1) + span / (param_nrs[wh] + 1), 1))
                  for wh in range(len(param_nrs))
                  for nr in range(param_nrs[wh])}
    else:
        bounds = {f"p{wh}{nr}": bounds_in
                  for wh in range(len(param_nrs))
                  for nr in range(param_nrs[wh])}
    aqc_obj.nshots = nshots

    def _objective(**params):
        return aqc_obj.run_protocol(path_list, params)

    optimizer = BayesianOptimization(f=_objective,
                                     pbounds=bounds,
                                     random_state=random_seed)
    # probe a linear path first
    initial_params = {f"p{wh}{nr}": (nr + 1) / (param_nrs[wh] + 1) for wh in range(len(param_nrs))
                      for nr in range(param_nrs[wh])}
    # set uncertainty of datapoints

    optimizer.set_gp_params(alpha=np.sqrt(1 / (nshots * aqc_obj.percentile)))

    optimizer.probe(initial_params, lazy=True)

    # this determines the point after which we start moving towards exploitation
    # (after decay_steps steps of optimization)
    decay_steps = int(num_iter * decay_ratio)
    if not decay_steps:
        decay_steps = 1
    # this determines the speed with which we decay the exploration rate.
    kappa_decay = 0.01 ** (1 / decay_steps)
    # this is the relevant parameter passed to the maximize function, telling when to start decaying exploration.
    kappa_decay_delay = int(num_iter - decay_steps)
    # optimize:
    optimizer.maximize(init_points=ip, n_iter=num_iter, acq=acq, xi=xi,
                       kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay)
    if return_target:
        # we return the final target value, i.e. the best overlap achieved during our optimization.
        return optimizer.max['target']
    else:
        # else we return the optimizer object.
        return optimizer


def random_parameters(param_nrs, t_total):
    """A function to generate random parameters in the correct dict form."""
    rand_params = {f"p{w}{n}": np.random.rand()
                   for w in range(len(param_nrs)) for n in range(param_nrs[w])}
    rand_params.update({'T': t_total})
    return rand_params


def linear_path_params(numpar, t_total, which=0):
    """Helper function that generates linear parameters for the real space parametrization."""
    lin_params = {f'p{which}{i}': (i + 1) / (numpar + 1) for i in range(numpar)}
    lin_params.update({'T': t_total})
    return lin_params


def optimize_fom1_get_fom2(fom1, fom2, aqc_obj, path_list, params):
    """Takes a BO object that has used energy to find the optimal parameters and then uses the fidelity as the
    figure of merit, to compare it to fidelity-based optimizers."""
    # change to overlap
    aqc_obj.fom = fom2
    # run protocol
    max_value_fom2 = aqc_obj.run_protocol(path_list, params)
    # change back
    aqc_obj.fom = fom1
    # return both in a list
    return max_value_fom2


# This file can be used to also generate data. The following executes if adquco.py is ran from the terminal:
def run_overlap_from_measurement(percentile, n_shots, random_seed):
    run_params = "ucb"
    if isinstance(run_params, str):
        acq = run_params
        xi = 0.05
        print('this evaluated')
    elif isinstance(run_params, tuple):
        acq, xi = run_params
    else:
        raise ValueError('Parameters specified the wrong way.')

    print(xi, ' is xi')
    # t_start = time.perf_counter()

    nqs = [10, 20, 30, 50, 75, 100, 125, 150]
    n_reps = 6

    int_steps = 500

    gamma = 5
    c = 0.8

    t_total = 50

    n_parameters = 6

    n_iter = 50
    n_init = 9

    # spans = [0.5, 1, 2]

    res_dict = {nq: {} for nq in nqs}

    for nq in nqs:
        print(f'nq = {nq}')
        res_dict[nq] = {typ: [] for typ in ['boqar-d', 'borar', 'boqaf-d', 'boraf', 'boqar-i', 'boqaf-i',
                                            'boqac-d', 'boqac-i', 'borac', 'lin', 'ra-lin']}
        # aqc object
        aqc = AQC([pspin(nq), gamma * v_tf(nq)], total_time=t_total, integration_steps=int_steps, percentile=percentile,
                  nshots=n_shots)
        lp = linear_path()
        linpaths = [lp, lambda t, _: 1 - lp(t, _)]

        aqc_r = AQC([pspin(nq, c), h_init(nq, c), gamma * v_tf(nq, c)], total_time=t_total, integration_steps=int_steps,
                    percentile=percentile, nshots=n_shots)
        ra_linpaths = [lp,
                       lambda t, _: (1 - lp(t, _)) ** 2,
                       lambda t, _: (1 - lp(t, _)) * lp(t, _)]

        for fom in ['measurements', 'overlap', 'energy']:
            aqc.fom = fom
            res_dict[nq]['lin'].append(aqc.run_protocol(linpaths, {}))
            aqc_r.fom = fom
            res_dict[nq]['ra-lin'].append(aqc_r.run_protocol(ra_linpaths, {}))
        # to have the correct figure of merit for training
        aqc.fom = 'measurements'
        aqc_r.fom = 'measurements'

        for seed in range(random_seed, random_seed + n_reps):
            print(f'{nq}; rep {seed - 42}')
            # dependent real (quantum annealing) boqa
            s = real_path(n_parameters, 0)
            dpaths = [s, lambda u, argument: (1 - s(u, argument))]
            boqar = bo_realspace(aqc, dpaths, [n_parameters], ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=2, return_target=False, acq=acq, xi=xi, nshots=n_shots)
            res_dict[nq]['boqar-d'].append(boqar.max['target'])
            res_dict[nq]['boqar-d'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc, dpaths, boqar))
            res_dict[nq]['boqar-d'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc, dpaths, boqar))

            # dependent cubic (quantum annealing) boqa
            s = real_path(n_parameters, 0, kind='cubic')
            dpaths = [s, lambda u, argument: (1 - s(u, argument))]
            boqac = bo_realspace(aqc, dpaths, [n_parameters], ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=2, return_target=False, acq=acq, xi=xi, nshots=n_shots)
            res_dict[nq]['boqac-d'].append(boqac.max['target'])
            res_dict[nq]['boqac-d'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc, dpaths, boqac))
            res_dict[nq]['boqac-d'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc, dpaths, boqac))

            # dependent fourier (quantum annealing) boqa
            s = fourier_path(n_parameters, 0)
            dpaths = [s, lambda u, argument: (1 - s(u, argument))]
            boqaf = bo_fourier(aqc, dpaths, [n_parameters], ip=n_init, num_iter=n_iter, random_seed=seed,
                               decay_ratio=0.5, acq=acq, xi=xi, return_target=False, nshots=n_shots)
            res_dict[nq]['boqaf-d'].append(boqaf.max['target'])
            res_dict[nq]['boqaf-d'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc, dpaths, boqaf))
            res_dict[nq]['boqaf-d'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc, dpaths, boqaf))

            # real independent:
            nps_i = [int(n_parameters / 2)] * 2
            s, lamb = tuple(real_path(pn, i) for i, pn in enumerate(nps_i))
            ipaths = [s, lambda u, argument: (1 - lamb(u, argument))]
            boqari = bo_realspace(aqc, ipaths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                  decay_ratio=0.5, acq=acq, xi=xi, return_target=False, span=2, nshots=n_shots)
            res_dict[nq]['boqar-i'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc, ipaths, boqari))
            res_dict[nq]['boqar-i'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc, ipaths, boqari))
            res_dict[nq]['boqar-i'].append(boqari.max['target'])

            # cubic independent:
            s, lamb = tuple(real_path(pn, i, kind='cubic') for i, pn in enumerate(nps_i))
            ipaths = [s, lambda u, argument: (1 - lamb(u, argument))]
            boqaci = bo_realspace(aqc, ipaths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                  decay_ratio=0.5, acq=acq, xi=xi, return_target=False, span=2, nshots=n_shots)
            res_dict[nq]['boqac-i'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc, ipaths, boqaci))
            res_dict[nq]['boqac-i'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc, ipaths, boqaci))
            res_dict[nq]['boqac-i'].append(boqaci.max['target'])

            # fourier independent
            s, lamb = tuple(fourier_path(pn, i) for i, pn in enumerate(nps_i))
            ipaths = [s, lambda u, argument: (1 - lamb(u, argument))]
            boqafi = bo_fourier(aqc, ipaths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                decay_ratio=0.5, acq=acq, xi=xi, return_target=False, nshots=n_shots)
            res_dict[nq]['boqaf-i'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc, ipaths, boqafi))
            res_dict[nq]['boqaf-i'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc, ipaths, boqafi))
            res_dict[nq]['boqaf-i'].append(boqafi.max['target'])

            # reverse annealing real (borar)
            nps_i = [int(n_parameters / 2)] * 2
            s, lamb = tuple(real_path(pn, i) for i, pn in enumerate(nps_i))
            paths = [s,
                     lambda u, argument: (1 - s(u, argument)) * (1 - lamb(u, argument)),
                     lambda u, argument: (1 - s(u, argument)) * lamb(u, argument)]

            borar = bo_realspace(aqc_r, paths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=1, acq=acq, xi=xi, return_target=False, nshots=n_shots)
            res_dict[nq]['borar'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc_r, paths, borar))
            res_dict[nq]['borar'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc_r, paths, borar))
            res_dict[nq]['borar'].append(borar.max['target'])

            # reverse annealing cubic (borac)
            s, lamb = tuple(real_path(pn, i, kind='cubic') for i, pn in enumerate(nps_i))
            paths = [s,
                     lambda u, argument: (1 - s(u, argument)) * (1 - lamb(u, argument)),
                     lambda u, argument: (1 - s(u, argument)) * lamb(u, argument)]

            borac = bo_realspace(aqc_r, paths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=1, acq=acq, xi=xi, return_target=False, nshots=n_shots)
            res_dict[nq]['borac'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc_r, paths, borac))
            res_dict[nq]['borac'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc_r, paths, borac))
            res_dict[nq]['borac'].append(borac.max['target'])

            # reverse annealing fourier (boraf)
            nps_i = [int(n_parameters / 2)] * 2
            s, lamb = tuple(fourier_path(pn, i) for i, pn in enumerate(nps_i))
            paths = [s,
                     lambda u, argument: (1 - s(u, argument)) * (1 - lamb(u, argument)),
                     lambda u, argument: (1 - s(u, argument)) * lamb(u, argument)]

            boraf = bo_fourier(aqc_r, paths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                               decay_ratio=0.5, acq=acq, xi=xi, return_target=False, nshots=n_shots)
            res_dict[nq]['boraf'].append(optimize_fom1_get_fom2('measurements', 'overlap', aqc_r, paths, boraf))
            res_dict[nq]['boraf'].append(optimize_fom1_get_fom2('measurements', 'energy', aqc_r, paths, boraf))
            res_dict[nq]['boraf'].append(boraf.max['target'])

    return res_dict
    # writing
    # with open(
    #         f"data/corrected-overlap_from_energy-{acq}{str(xi).replace('.', '_') if acq == 'ei' else ''}.json",
    #         'w'
    # ) as f:
    #     f.write(dumps(res_dict))

    # print('Time elapsed: {:.2f}s'.format(time.perf_counter() - t_start))


def run_overlap(random_seed):
    run_params = "ucb"
    if isinstance(run_params, str):
        acq = run_params
        xi = 0.05
        print('this evaluated')
    elif isinstance(run_params, tuple):
        acq, xi = run_params
    else:
        raise ValueError('Parameters specified the wrong way.')

    print(xi, ' is xi')
    # t_start = time.perf_counter()

    nqs = [10, 20, 30, 50, 75, 100, 125, 150]
    n_reps = 1

    int_steps = 500

    gamma = 5
    c = 0.8

    t_total = 50

    n_parameters = 6

    n_iter = 50
    n_init = 9

    # spans = [0.5, 1, 2]

    res_dict = {nq: {} for nq in nqs}

    for nq in nqs:
        print(f'nq = {nq}')
        res_dict[nq] = {typ: [] for typ in ['boqar-d', 'borar', 'boqaf-d', 'boraf', 'boqar-i', 'boqaf-i',
                                            'boqac-d', 'boqac-i', 'borac']}
        # aqc object
        aqc = AQC([pspin(nq), gamma * v_tf(nq)], total_time=t_total, integration_steps=int_steps, fom='overlap')
        lp = linear_path()
        linear_overlap = aqc.run_protocol([lp, lambda t, _: 1 - lp(t, _)], {})
        res_dict[nq]['lin'] = linear_overlap

        for seed in range(random_seed, random_seed + n_reps):
            print(f'{nq}; rep {seed - 42}')
            # dependent real (quantum annealing) boqa
            s = real_path(n_parameters, 0)
            dpaths = [s, lambda u, argument: (1 - s(u, argument))]
            boqar = bo_realspace(aqc, dpaths, [n_parameters], ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=2, return_target=True, acq=acq, xi=xi)
            res_dict[nq]['boqar-d'].append(boqar)
            # cubic real
            s = real_path(n_parameters, 0, kind='cubic')
            dpaths = [s, lambda u, argument: (1 - s(u, argument))]
            boqac = bo_realspace(aqc, dpaths, [n_parameters], ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=2, return_target=True, acq=acq, xi=xi)
            res_dict[nq]['boqac-d'].append(boqac)

            # dependent fourier (quantum annealing) boqa
            s = fourier_path(n_parameters, 0)
            dpaths = [s, lambda u, argument: (1 - s(u, argument))]
            boqaf = bo_fourier(aqc, dpaths, [n_parameters], ip=n_init, num_iter=n_iter, random_seed=seed,
                               decay_ratio=0.5, acq=acq, xi=xi, return_target=True)
            res_dict[nq]['boqaf-d'].append(boqaf)
            # real independent:
            nps_i = [int(n_parameters / 2)] * 2
            s, lamb = tuple(real_path(pn, i) for i, pn in enumerate(nps_i))
            ipaths = [s, lambda u, argument: (1 - lamb(u, argument))]
            boqari = bo_realspace(aqc, ipaths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                  decay_ratio=0.5, acq=acq, xi=xi, return_target=True, span=2)
            res_dict[nq]['boqar-i'].append(boqari)
            # cubic independent:
            s, lamb = tuple(real_path(pn, i, kind='cubic') for i, pn in enumerate(nps_i))
            ipaths = [s, lambda u, argument: (1 - lamb(u, argument))]
            boqaci = bo_realspace(aqc, ipaths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                  decay_ratio=0.5, acq=acq, xi=xi, return_target=True, span=2)
            res_dict[nq]['boqac-i'].append(boqaci)
            # fourier independent
            s, lamb = tuple(fourier_path(pn, i) for i, pn in enumerate(nps_i))
            ipaths = [s, lambda u, argument: (1 - lamb(u, argument))]
            boqafi = bo_fourier(aqc, ipaths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                decay_ratio=0.5, acq=acq, xi=xi, return_target=True)
            res_dict[nq]['boqaf-i'].append(boqafi)
            # reverse annealing real (borar)
            s, lamb = tuple(real_path(pn, i) for i, pn in enumerate(nps_i))
            paths = [s,
                     lambda u, argument: (1 - s(u, argument)) * (1 - lamb(u, argument)),
                     lambda u, argument: (1 - s(u, argument)) * lamb(u, argument)]

            aqc_r = AQC([pspin(nq, c), h_init(nq, c), gamma * v_tf(nq, c)], total_time=t_total, fom='overlap',
                        integration_steps=int_steps)

            borar = bo_realspace(aqc_r, paths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=1, acq=acq, xi=xi, return_target=True)
            res_dict[nq]['borar'].append(borar)
            # reverse cubic annealing (borac)
            s, lamb = tuple(real_path(pn, i, kind='cubic') for i, pn in enumerate(nps_i))
            paths = [s,
                     lambda u, argument: (1 - s(u, argument)) * (1 - lamb(u, argument)),
                     lambda u, argument: (1 - s(u, argument)) * lamb(u, argument)]

            borac = bo_realspace(aqc_r, paths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                                 decay_ratio=0.5, span=1, acq=acq, xi=xi, return_target=True)
            res_dict[nq]['borac'].append(borac)
            # reverse annealing fourier (boraf)
            s, lamb = tuple(fourier_path(pn, i) for i, pn in enumerate(nps_i))
            paths = [s,
                     lambda u, argument: (1 - s(u, argument)) * (1 - lamb(u, argument)),
                     lambda u, argument: (1 - s(u, argument)) * lamb(u, argument)]

            boraf = bo_fourier(aqc_r, paths, nps_i, ip=n_init, num_iter=n_iter, random_seed=seed,
                               decay_ratio=0.5, acq=acq, xi=xi, return_target=True)
            res_dict[nq]['boraf'].append(boraf)

        linear_overlap_ra = aqc_r.run_protocol([lp,
                                                lambda t, _: (1 - lp(t, _)) ** 2,
                                                lambda t, _: (1 - lp(t, _)) * lp(t, _)],
                                               {})
        res_dict[nq]['ra-lin'] = linear_overlap_ra
    return res_dict


def overlap_from_measurements_mappable(arguments):
    return run_overlap_from_measurement(**arguments)


if __name__ == "__main__":
    import multiprocessing
    from itertools import product

    # import sys
    nseeds = 25
    nreps = 6
    seeds = range(420, 420 + nseeds * nreps, nreps)

    percentiles_pars = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

    nshots_pars = [500]

    map_arguments = [dict(zip(['random_seed', 'percentile', 'n_shots'], pars))
                     for pars in product(seeds, percentiles_pars, nshots_pars)]
    # map_arguments = list(product(percentiles_pars, nshots_pars, seeds))
    print(len(map_arguments))
    print(map_arguments)

    with multiprocessing.Pool(processes=len(map_arguments)) as p:
        out = p.map(overlap_from_measurements_mappable, map_arguments)
    for perc, nshot in product(percentiles_pars, nshots_pars):
        master_dict = {}
        for nqub in out[0].keys():
            master_dict[nqub] = {k: [] for k in ['boqar-d', 'borar', 'boqaf-d', 'boraf', 'boqar-i', 'boqaf-i',
                                                 'boqac-d', 'boqac-i', 'borac']}
            for position, d in enumerate(out):
                if map_arguments[position]["percentile"] == perc and map_arguments[position]["n_shots"] == nshot:
                    for k in master_dict[nqub].keys():
                        master_dict[nqub][k] += d[nqub][k]
                    master_dict[nqub].update({'lin': out[position][nqub]['lin'],
                                              "ra-lin": out[position][nqub]['ra-lin']})

        with open(f"data/pspin/additional_overlap_from_measurements-perc{perc}-nsh{nshot}.json", 'w') as f:
            f.write(dumps(master_dict))
    # acqs = ['ucb', ('ei', 0.01), ('ei', 0.05), ('ei', 0.1)]
    # with multiprocessing.Pool(len(acqs)) as p:
    #     p.map(run, acqs)
