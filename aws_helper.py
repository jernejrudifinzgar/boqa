import os
import numpy as np

from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.jobs.metrics import log_metric
from braket.jobs import save_job_checkpoint

from itertools import combinations, chain
import subprocess
import sys

# If you don't build the image on aws yourself, you can do this quick & dirty fix:
try:
    from bayes_opt import BayesianOptimization, UtilityFunction
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bayesian-optimization'])
    from bayes_opt import BayesianOptimization, UtilityFunction

from scipy.spatial import distance_matrix
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

import networkx as nx
import json
from itertools import product


# noinspection PyTypeChecker
def run_job(hyperparams=None):
    """Main function which is ran in the hybrid jobs. If hyperparameters are passed, it is ran locally. Else, it reads them
    from the environment variables."""

    is_local = hyperparams is not None
    print("Job started!!!!!")

    # load hyperparameters
    if not is_local:
        hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
        input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
        with open(hp_file, "r") as f:
            hyperparams = json.load(f)
    else:
        input_dir = ""

    # convert hyperparameters to variables of correct type
    # number of shots
    n_shots = int(hyperparams["n_shots"])
    # number of detuning points
    ndp = int(hyperparams["ndp"])
    # number of initial points for BO to take
    n_initialpoints = int(hyperparams["n_initialpoints"])
    # number of iterations for BO to take
    n_iterations = int(hyperparams["n_iterations"])
    # which graph type / index is chosen
    graph_parameter = hyperparams["graph_parameter"]
    graph_kind, graph_idx = graph_parameter[0], int(graph_parameter[1:])
    # percentile of energies used for the cost function
    percent = float(hyperparams["percentile"])
    # type of drive
    drivetype = hyperparams["drivetype"]
    # scale of the lattice, multiplied by 4.5\micro\meter
    lattice_scale = float(hyperparams["lattice_scale"])
    # alpha for the classical cost function
    alpha = float(hyperparams["alpha"]) if "alpha" in hyperparams else 5
    # do we postselect on the correct initialization of the atoms or not?
    postselection = bool(int(hyperparams["postselection"])) if "postselection" in hyperparams.keys() else False
    # total time of the protocol
    t_total = np.round(float(hyperparams['t_total']) if "t_total" in hyperparams.keys() else 4e-6, 9)
    # do we optimize the rabi frequency path or not?
    optimize_rabi = True if "optimize_rabi" not in hyperparams.keys() else bool(int(hyperparams['optimize_rabi']))

    # check for checkpoints -- this enables us to restart jobs that weren't able to complete within the time limit
    checkpoint_data = None
    if not is_local:
        checkpoints_path = os.environ["AMZN_BRAKET_CHECKPOINT_DIR"]
        checkpoint_file = ''
        print(checkpoints_path)
        print(os.listdir(checkpoints_path))
        for item in os.listdir(checkpoints_path):
            if item.endswith('checkpoint.json'):
                print('Checkpoint file found')
                checkpoint_file = os.path.join(checkpoints_path, item)
                break
        is_checkpoint = checkpoint_file != ''
        if is_checkpoint:
            with open(os.path.join(checkpoints_path, checkpoint_file), 'r') as f:
                checkpoint_data = json.load(f)

            print(checkpoint_data)
            print("THIS WORKED!!!!")

    if not optimize_rabi:
        rabi_params = {"omega_max": 1.58e7, "tau": 0.1}
    else:
        rabi_params = {}

    # Use the device declared in the job script
    if is_local:
        device = LocalSimulator('braket_ahs')
    else:
        device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])
        # device = LocalSimulator('braket_ahs')

    # preparation of the register and graph
    if graph_kind == 't':
        register, graph = prepare_test_register(n_cells=graph_idx, parallel=not is_local, a=4.5e-6 * lattice_scale)
    elif graph_kind == 'h':
        register, graph = prepare_register_hard_graph(
            os.path.join(input_dir, "hard-graphs-13-14/", f"hg{graph_idx}.txt"), scale=lattice_scale)
    elif graph_kind == 'g':
        register, graph = prepare_grid_register(which=graph_idx, parallel=not is_local, a=4.5e-6 * lattice_scale)
    elif graph_kind == "e":
        register, graph = prepare_register_hard_graph(
            os.path.join(input_dir, "easy-graphs/", f"eg{graph_idx}.txt"), scale=lattice_scale)
    else:
        raise Exception(f"Graph kind {graph_kind} not recognized.")
    print("Register generated!")
    
    # determine the directory where the results are saved
    if is_local:
        dir_path = ""
    else:
        dir_path = os.environ['AMZN_BRAKET_JOB_RESULTS_DIR']

    # save graph and vertex positions
    if not is_local:
        nx.write_adjlist(graph, os.path.join(dir_path, "graph.adjlist"))
        np.savetxt(fname=os.path.join(dir_path, "atom_positions.txt"),
                   X=np.transpose([register.coordinate_list(i) for i in [0, 1]]))

    def _objective_with_data(params, shot_factor=1, percentile=1.):
        """Objective function, that also returns the measured bitsrings. This is used for the classical cost function."""
        # prepares the Hamiltonian, given the parameters.
        H_obj = prepare_drive(t_total, params, kind=drivetype)
        # defines the ahs_program object which is sent o the device.
        ahs_program = AnalogHamiltonianSimulation(hamiltonian=H_obj, register=register)
        ns = int(shot_factor * n_shots)
        result = device.run(ahs_program, shots=ns).result()
        # extracts bitstrings (as a list of lists -- actually a 2D numpy array)
        if postselection:
            measured_bistrings = np.array([np.array(s.post_sequence) for s in result.measurements
                                           if np.all(s.pre_sequence)])
        else:
            measured_bistrings = np.array([np.array(s.post_sequence) for s in result.measurements])
        # returns the MIS cost and the bitstrings. We make sure that the number of shots considered for the optimizer is
        # the same.
        energies = [get_mis_cost(graph, bs, alpha=alpha) for bs in measured_bistrings]
        # we only consider the lowest percentile of the energies
        included = np.sort(energies)[:int(percentile * len(energies))]
        return -np.mean(included), measured_bistrings

    # This sets the parametrization parameter for the toy graphs (which are not used in the paper.)
    if graph_kind == 't':
        if graph_idx <= 3 and not is_local:
            n_parallel = (5 - graph_idx) ** 2
        else:
            n_parallel = 1
    else:
        n_parallel = 1

    # parameter bounds, optimizer initialization and tuning
    pbounds = get_parameter_bounds(ndp, span=2, kind=drivetype, scale=lattice_scale, optimize_rabi=optimize_rabi)
    # initialize the optimizer
    optimizer = BayesianOptimization(f=None, pbounds=pbounds)
    # we set the alpha parameter of the GP to be inversely proportional to the number of shots, to account for the fact
    # that the variance of the MIS cost is proportional to the number of shots.
    if postselection:
        optimizer.set_gp_params(alpha=graph.number_of_nodes() / (n_parallel * n_shots * 0.99 ** n_shots))
    else:
        optimizer.set_gp_params(alpha=graph.number_of_nodes() / (n_parallel * n_shots))

    # loading checkpoint data
    if checkpoint_data is not None:
        print("Loading checkpoint data:")
        checkpoint_data_dict = checkpoint_data['dataDictionary']
        already_performed = len(checkpoint_data_dict)
        print(checkpoint_data_dict)
        print(f"already performed {already_performed} iterations")
        for datapoint in checkpoint_data_dict.values():
            pars = {p: v for p, v in datapoint["params"].items() if p in pbounds.keys()}
            optimizer.register(pars, datapoint["target"])
        print("checkpoint data loaded.. this worked!!!")
        print(optimizer.res)
        # set the correct number of iterations to still perform:
        if already_performed <= n_initialpoints:
            # if only initialpoints have been performed:
            previously_done_initialpoints = already_performed
            previously_done_iterations = 0
        else:
            # if also some of the iterations have been performed, we need to see how many:
            previously_done_initialpoints = n_initialpoints
            # no initial points are required then
            previously_done_iterations = already_performed - n_initialpoints
    else:
        previously_done_initialpoints = previously_done_iterations = 0
    # results saving dictionary parameters:
    res_dict = {}
    resdict_filepath = os.path.join(dir_path, "optimizer_results.json")

    # shot factor for initial and best data:
    sf = 8
    # register initial params & initialize iteration counter.
    initial_params = generate_initial_params(omega_max=np.mean(compute_rabi_bounds(lattice_scale * 4.5e-6)),
                                             delta_initial=3e7, delta_final=6e7, tau=1 / 10, ndp=ndp, kind=drivetype)
    initial_params.update(rabi_params)
    initial_target, initial_data = _objective_with_data(initial_params, shot_factor=sf, percentile=percent)

    if checkpoint_data is not None:
        # we set the maxima accordingly.
        res_dict = {i: rd for i, rd in enumerate(optimizer.res)}
        tmp_targets = [rd['target'] for rd in optimizer.res]
        best_target = np.max(tmp_targets)
        best_params = optimizer.res[np.argmax(tmp_targets)]['params']
        best_params.update(rabi_params)
        best_data = []

    else:
        # else we proceed normally
        res_dict = add_save_optimizer_data(res_dict, initial_params, initial_target, resdict_filepath, is_local)
        optimizer.register(target=initial_target, params=keep_pbound_keys(initial_params, pbounds))
        best_target = initial_target
        best_data = initial_data
        best_params = initial_params
    print(res_dict)
    if not is_local:
        # save initial data:
        # noinspection PyTypeChecker
        np.savetxt(fname=os.path.join(dir_path, "initial_bitstrings.txt"), X=initial_data)

    # run the initial points.
    for _ in range(previously_done_initialpoints, n_initialpoints):
        rp = suggest_random_params(pbounds)
        rp.update(rabi_params)
        target, data = _objective_with_data(rp, percentile=percent)
        # save data if target is best so far
        if target > best_target:
            best_data = data
            best_target = target
            best_params = rp
        # save results (checkpoint)
        res_dict = add_save_optimizer_data(res_dict, rp, target, resdict_filepath, is_local)
        optimizer.register(params=keep_pbound_keys(rp, pbounds), target=target)

    # utility function parameters
    acq = 'ucb'
    decay_delay_ratio = 0.5
    decay_delay = round(decay_delay_ratio * n_iterations)
    decay_steps = n_iterations - decay_delay

    def _kappa(st, kappa_initial, kappa_final):
        """Returns the value of kappa at step st."""
        if step < decay_delay:
            return kappa_initial
        else:
            decay_power_step = st - decay_delay
            return kappa_initial * (kappa_final / kappa_initial) ** (decay_power_step / decay_steps)

    # initial and final values of kappa
    ki, kf = 2, 0.01

    # check if there were perhaps some iterations done beforehand.
    for step in range(previously_done_iterations, n_iterations):
        kap = _kappa(step, ki, kf)
        utility = UtilityFunction(kind=acq, kappa=kap, xi=0.)
        next_params = optimizer.suggest(utility)
        next_params.update(rabi_params)
        print(f"Point suggested; kappa={kap:.3f}.")

        target, data = _objective_with_data(next_params, percentile=percent)
        if target > best_target:
            best_data = data
            best_target = target
            best_params = next_params
        res_dict = add_save_optimizer_data(res_dict, next_params, target, resdict_filepath, is_local)
        optimizer.register(params=keep_pbound_keys(next_params, pbounds), target=target)

    # we want better statistics of the best obtained parameters; to compare with initial solution.
    if (sf > 1) or (best_data == []):
        if best_data == []:
            sf += 1
        _, more_best_data = _objective_with_data(best_params, shot_factor=sf, percentile=percent)
        if best_data == []:
            best_data = more_best_data
        else:
            best_data = np.vstack((best_data, more_best_data))

    if not is_local:
        # save best data:
        np.savetxt(fname=os.path.join(dir_path, "best_data.txt"), X=best_data)
        with open(os.path.join(dir_path, "hyperparameters.json"), 'w') as fhyper:
            fhyper.write(json.dumps(hyperparams))
    print("Job completed!")
    if is_local:
        return optimizer, initial_data, best_data


def keep_pbound_keys(parameters, pbounds):
    """Keeps only the keys of parameters that are also in pbounds."""
    return {k: v for k, v in parameters.items() if k in pbounds}


def prepare_register_hard_graph(fpath, scale=1):
    """Takes a filepath to a txt file containing the positions of vertices and returns the register and the graph
    object. scale=1 corresponds to a lattice constant of 4.5."""
    vs = np.loadtxt(fpath) * scale
    vs = vs.round(7)
    reg = AtomArrangement()
    for v in vs:
        reg.add(v)
    return reg, vertices_to_graph(vs, 1.7 * scale * 4.5e-6)


def add_save_optimizer_data(res_dict, params, target_value, filepath, is_local):
    """Adds optimizer data to the dictionary saving the results. Also writes it to filepath."""
    pos = len(res_dict)
    new_entry = {'params': params, 'target': target_value}
    res_dict[pos] = new_entry
    # print out for progress tracking!
    print(f"Iteration {pos + 1} completed!")
    if not is_local:
        with open(filepath, 'w') as f_resdict:
            f_resdict.write(json.dumps(res_dict))
        # for logging
        log_metric(metric_name="Target", value=target_value, iteration_number=pos + 1)
        # for checkpointing
        save_job_checkpoint(checkpoint_data=res_dict,
                            checkpoint_file_suffix="checkpoint")

    return res_dict


def log_iteration(iteration_number):
    print(f"Iteration {iteration_number} completed.")
    iteration_number += 1
    return iteration_number


def suggest_random_params(pbounds):
    """Suggests random params."""
    return {k: np.random.uniform(*v) for k, v in pbounds.items()}


def get_mis_cost(graph, solution, alpha=5):
    """Calculate the mis cost for a give solution for a given graph."""
    solution = np.array(solution)
    violations = 0
    excited_pos = np.where(solution == 0)[0]
    for v1, v2 in graph.edges:
        if not solution[v1] and not solution[v2]:
            violations += 1
    return -len(excited_pos) + alpha * violations


def get_parameter_bounds(ndp, span=1, kind='real', scale=1., optimize_rabi=True):
    """Get the bounds for the BO."""
    omega_bounds = tuple(sorted(compute_rabi_bounds(4.5e-6 * scale)))
    pb = {
        "omega_max": omega_bounds,
        "delta_initial": (1e7, 1.25e8),
        "delta_final": (1e7, 1.25e8),
        "tau": (0.05, 0.45)
    }
    if not optimize_rabi:
        del pb["omega_max"]
        del pb['tau']

    if kind != 'fourier':
        pb.update({f"dp{nr + 1}": (
            max((nr + 1) / (ndp + 1) - span / (ndp + 1), 0),
            min((nr + 1) / (ndp + 1) + span / (ndp + 1), 1)
        ) for nr in range(ndp)})
    else:
        pb.update({f"dp{nr + 1}": (-1 / (nr + 1), 1 / (nr + 1)) for nr in range(ndp)})

    return pb


def generate_initial_params(omega_max, delta_initial, delta_final, tau, ndp, kind='real'):
    """Generate initial parameters for the optimization -- takes the arguments of the function and returns a dictionary
    with the parameters in the correct format."""
    params = {
        'omega_max': omega_max,
        'delta_initial': delta_initial,
        'delta_final': delta_final,
        'tau': tau
    }
    if kind != 'fourier':
        params.update({f'dp{i}': i / (ndp + 1) for i in range(1, ndp + 1)})
    else:
        params.update({f'dp{i}': 0 for i in range(1, ndp + 1)})

    return params


def prepare_test_register(n_cells=1, parallel=False, a=4.5e-6, add_middle=True):
    """Prepares a register for testing -- a grid of atoms with a middle atom, which was used for
    characterization of the system."""
    # a = 4.5e-6
    register = AtomArrangement()
    if not parallel:
        n_reps = 1
    else:
        n_reps = int(np.floor((7.5e-5 / (2 * a) + 1) / (n_cells + 1)))

    for offset in product(range(n_reps), range(n_reps)):
        x_o, y_o = [o * 2 * a * (n_cells + 1) for o in offset]
        # grid
        for i in range(0, 2 * n_cells + 1, 2):
            for j in range(0, 2 * n_cells + 1, 2):
                register.add(np.array((i * a + x_o, j * a + y_o)).round(7))
        # add middle
        if add_middle:
            for i in range(1, 2 * n_cells + 1, 2):
                for j in range(1, 2 * n_cells + 1, 2):
                    register.add(np.array((i * a + x_o, j * a + y_o)).round(7))

    return register, vertices_to_graph(np.transpose([register.coordinate_list(0), register.coordinate_list(1)]),
                                       a * 1.5)


def prepare_grid_register(which=0, parallel=False, a=4.5e-6):
    """Prepares the register of the small grid graphs shown in the paper. Which is the index of the graph to be used,
    and runs between 0 and 10. It also derives the correct parallelization scheme."""
    # number of grid points in the x, y dimension
    mx = 4
    my = 3

    def _get_reps_and_shift(m, d=7.5e-5):
        reps = int((d + 2 * a) / (a * (m + 2)))
        # print(reps)
        shift = (d - reps * a * m) / ((reps - 1) * a)
        return reps, shift

    if not parallel:
        x_reps = y_reps = 1
        x_shift = y_shift = 0
    else:
        x_reps, x_shift = _get_reps_and_shift(mx - 1, d=7.5e-5)
        y_reps, y_shift = _get_reps_and_shift(my - 1, d=7.6e-5)

    _, poss = get_grid_graph_combs(9, 10, (mx, my), scale=a)
    if which >= len(poss):
        print("Warning; which is longer than the number of graphs generated.")
        which %= len(poss)
        print(f"Graph {which} generated.")
    pos = poss[which]

    pos_container = []

    for xrep in range(x_reps):
        for yrep in range(y_reps):
            pos_container.append(pos + np.array([xrep * (mx + x_shift - 1) * a, yrep * (my + y_shift - 1) * a]))

    total_pos = np.vstack(pos_container)

    register = AtomArrangement()
    for coord in total_pos:
        register.add(coord.round(7))

    return register, vertices_to_graph(total_pos, 1.5 * a)


def prepare_drive(total_time, params, kind='real'):
    """Wrapper for the drive parametrization functions."""
    H = Hamiltonian()

    # this function does the heavy lifting:
    if kind == 'real':
        drive = get_drive_real(total_time, params, lowpass=False)
    elif kind == "lowpass":
        drive = get_drive_real(total_time, params, lowpass=True)
    elif kind == "fourier":
        drive = get_drive_real(total_time, params)
    else:
        raise NotImplementedError("Kind of drive parametrization not implemented.")
    H += drive

    return H


def cutoff_ramp(t, total_time, tau):
    """Cutoff ramp function."""
    if t <= tau:
        return 0
    elif t >= total_time - tau:
        return 1
    else:
        return (t - tau) / (total_time - 2 * tau)


def get_drive_fourier(total_time, params):
    """Fourier drive."""
    Omega, phi = get_omega_phi_ts(total_time, params)
    delta_initial = round(params['delta_initial'])
    delta_final = round(params['delta_final'])
    coefs = {k: float(v) for (k, v) in params.items() if k.startswith('dp')}
    tau = params["tau"] * total_time
    interval = total_time - 2 * tau
    nts = int(total_time / 5e-8)
    times = [0] + list(np.linspace(tau, total_time - tau, nts)) + [total_time]

    def _sinusoid(t, freq_idx, magnitude):
        if t <= tau or t >= total_time - tau:
            return 0
        else:
            freq = freq_idx * np.pi / interval
            return np.sin(freq * (t - tau)) * magnitude

    # linear ramp
    vals = np.array([cutoff_ramp(t, total_time, tau) for t in times])
    for dpfi, coef in coefs.items():
        fi = int(dpfi[2:])
        vals += np.array([_sinusoid(t, fi, coef) for t in times])
    deltas = np.clip(-delta_initial + (delta_final + delta_initial) * vals, -1.25e8, 1.25e8)

    Delta_global = TimeSeries()
    for t, delta in zip(times, deltas):
        Delta_global.put(round(t, 9), round(delta))
    return DrivingField(amplitude=Omega, phase=phi, detuning=Delta_global)


def get_real_detuning_ts(params, total_time):
    """Computes the detuning schedule for a real-point schedule. No Low-Pass filtering."""
    delta_initial = round(params['delta_initial'])
    delta_final = round(params['delta_final'])
    delta_points = {k: float(v) for (k, v) in params.items() if k.startswith('dp')}
    ndp = len(delta_points)
    tau = round(params["tau"] * total_time, 9)
    # start with negative detuning
    Delta_global = TimeSeries().put(0.0, -delta_initial)
    Delta_global.put(tau, -delta_initial)
    # add intermediate points
    for i in range(1, ndp + 1):
        # correctly stretch the points in between, as they are dependent on delta_initial and delta_final
        d_val = round(-delta_initial + (delta_final - (-delta_initial)) * delta_points[f'dp{i}'])
        Delta_global.put(round(tau + (total_time - 2 * tau) * i / (ndp + 1), 9), d_val)
    Delta_global.put(round(total_time - tau, 9), delta_final)
    # add a positive delta_final
    Delta_global.put(round(total_time, 9), delta_final)  # (time [s], value [rad/s])

    return Delta_global


def get_lowpass_detuning_ts(params, total_time):
    """Computes the detuning schedule for a real-point schedule. Low-Pass filtering."""
    def _butter_lowpass(cutOff, fs, order=5):
        nyq = 0.5 * fs
        normalCutoff = cutOff / nyq
        # noinspection PyTupleAssignmentBalance
        b, a = butter(order, normalCutoff, btype='low', analog=False, output='ba')
        return b, a

    def _butter_lowpass_filter(data, cutOff, fs, order=4):
        b, a = _butter_lowpass(cutOff, fs, order=order)
        ys = lfilter(b, a, data)
        return ys

    delta_initial = round(params['delta_initial'])
    delta_final = round(params['delta_final'])
    delta_points = {k: float(v) for (k, v) in params.items() if k.startswith('dp')}
    tau = round(params["tau"] * total_time, 9)
    points = [0, 0] + [delta_points[k] for k in sorted(delta_points.keys())] + [1, 1]
    interp_times = [0, tau] + list(np.linspace(tau, total_time - tau, len(delta_points) + 2)[1:-1]) + \
                   [total_time - tau, total_time]
    interp = interp1d(interp_times, points)
    nts = int(total_time / 5e-8)
    times = list(np.linspace(0, total_time, nts))

    ramp = np.array([cutoff_ramp(t, total_time, tau) for t in times])
    series_notrend = interp(times) - ramp
    y = _butter_lowpass_filter(series_notrend, nts // (len(points) + 1), nts, 3) + ramp

    Delta_global = TimeSeries()
    for t, pt in zip(times, y):
        delta = int(np.round(np.clip(-delta_initial + (delta_final - (-delta_initial)) * pt, -1.25e8, 1.25e8)))
        Delta_global.put(round(t, 9), delta)

    return Delta_global


def get_omega_phi_ts(total_time, params):
    """Computes the Rabi frequency and phase schedules."""
    Omega_max = (params['omega_max'] // 400) * 400  # rad / seconds. We're fixing the resolution to 400.
    tau = params['tau']  # fraction of total time

    # Rabi frequency path
    Omega = TimeSeries()
    # first linear ramp up
    Omega.put(0.0, 0.)
    Omega.put(round(tau * total_time, 9), Omega_max)
    # constant for (1 - 2 * tau) * total_time
    Omega.put(round(total_time * (1 - tau), 9), Omega_max)
    # ramp down
    Omega.put(round(total_time, 9), 0.)
    # no phase (i.e. 0 the whole time)
    phi = TimeSeries().put(0.0, 0.0).put(total_time, 0.0)  # (time [s], value [rad])
    return Omega, phi


def get_drive_real(total_time, params, lowpass=False):
    """Returns a DrivingField object for a real-point schedule. This is used for the real experiment."""
    Omega, phi = get_omega_phi_ts(total_time, params)
    if not lowpass:
        Delta_global = get_real_detuning_ts(params, total_time)
    else:
        Delta_global = get_lowpass_detuning_ts(params, total_time)

    drive = DrivingField(
        amplitude=Omega,
        phase=phi,
        detuning=Delta_global
    )
    return drive


def get_grid_graph_combs(natoms_min, natoms_max, gridshape, scale=5.2e-6, return_combs=False):
    """Returns all different graphs of sizes natoms_min to natoms_max which have a unique MIS. The atoms are positioned
    in a grid of shape gridshape. The distance between atoms is scale."""
    poss = []
    graphs = []
    good_combs = []

    for natoms in range(natoms_min, natoms_max + 1):
        for comb in combinations(range(np.multiply(*gridshape)), natoms):
            indices = np.unravel_index(comb, gridshape)
            pos = np.transpose(indices) * scale
            g = vertices_to_graph(pos)
            _, miss = find_mis(g)
            if len(miss) == 1 and not np.any([nx.is_isomorphic(g, graph) for graph in graphs]):
                graphs.append(g)
                poss.append(pos)
                good_combs.append(comb)

    if return_combs:
        return good_combs
    else:
        return graphs, poss


def powerset(iterable):
    """Returns the powerset of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, q) for q in range(len(s) + 1))


def is_independent(graph, solution):
    """Checks if a given solution is independent."""
    for edge in product(solution, repeat=2):
        if graph.has_edge(*edge):
            return False
    return True


def find_mis(graph, maximum=True):
    """Finds a maximal independent set of a graph, and returns its bitstrings."""
    n_nodes = graph.number_of_nodes()
    if maximum is False:
        colored_nodes = nx.maximal_independent_set(graph)
        return len(colored_nodes), colored_nodes
    else:
        solutions = []
        maximum = 0
        for subset in powerset(range(n_nodes)):
            if is_independent(graph, subset):
                if len(subset) > maximum:
                    solutions = [subset]
                    maximum = len(subset)
                elif len(subset) == maximum:
                    solutions.append(subset)
        return maximum, solutions


def vertices_to_graph(vertices, radius=7.5e-6):
    """Converts the positions of vertices into a UDG."""
    dmat = distance_matrix(vertices, vertices)
    adj = (dmat < radius).astype(int) - np.eye(len(vertices))
    return nx.from_numpy_array(adj)


def compute_rabi_bounds(lattice_spacing):
    """Computes the (approximate) bounds on the allowed Rabi frequencies."""
    c6 = 5.42e-24
    return min(c6 / (lattice_spacing * np.sqrt(2)) ** 6, 1.58e7), min(c6 / (lattice_spacing * 2) ** 6, 1.58e7)


# noinspection PyTypeChecker
def test_spacing():
    """Test how spacing and omega influence the defects in the middle of the array."""
    device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])
    dir_path = os.environ['AMZN_BRAKET_JOB_RESULTS_DIR']
    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]

    with open(hp_file, "r") as f:
        hyperparams = json.load(f)

    empty = hyperparams["which"] == "empty"
    nshots = 500
    t_total = 4e-6
    a_list = np.linspace(4.2e-6, 6.2e-6, 4)

    for spacing in a_list:
        register, _ = prepare_test_register(n_cells=6, parallel=False, a=spacing, add_middle=not empty)
        omega_min, omega_max = compute_rabi_bounds(spacing)
        if not empty:
            omegas = np.linspace(omega_min, omega_max, 4)
        else:
            omegas = [np.mean([omega_min, omega_max])]

        for omega in omegas:
            params = generate_initial_params(omega_max=omega, delta_initial=3e7,
                                             delta_final=6e7, tau=1 / 10, ndp=0, kind='real')
            H_obj = prepare_drive(t_total, params, kind='real')
            # defines the ahs_program object which is sent o the device.
            ahs_program = AnalogHamiltonianSimulation(hamiltonian=H_obj, register=register)
            result = device.run(ahs_program, shots=nshots).result()
            # extracts bistrings (as a list of lists)
            measured_bistrings = np.array([np.array(s.post_sequence) for s in result.measurements])
            np.savetxt(fname=os.path.join(dir_path, f"bs-a{spacing * 1e6:.2f}-om{omega / 1e6:.3f}.txt"),
                       X=measured_bistrings)


def run_gridgraph_parameters(hyperparams=None):
    """Reads in the optimal parameters obtained in classical simulations and runs them on the quantum
    device."""
    is_local = hyperparams is not None
    if not is_local:
        hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
        input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]

        with open(hp_file, "r") as f:
            hyperparams = json.load(f)
        dir_path = os.environ['AMZN_BRAKET_JOB_RESULTS_DIR']
        device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])
        # device = LocalSimulator('braket_ahs')

    else:
        input_dir = ""
        dir_path = ""
        device = LocalSimulator('braket_ahs')

    t_total = float(hyperparams["t_total"]) if "t_total" in hyperparams else 4e-6
    n_shots = int(hyperparams["n_shots"]) if "n_shots" in hyperparams else 60
    which_params = int(hyperparams["which_params"]) if "which_params" in hyperparams else 0

    with open(os.path.join(input_dir, "gridgraph-params/", f"pretrained_params-{which_params}.json"), 'r') as f:
        params = json.load(f)

    param_names = ["delta_initial", "delta_final", "dp1", "dp2"]

    bitstrings_dict = {}

    for param_nr, data in params.items():
        graph_idx = int(data['graph'])
        param_nr = int(param_nr)
        register, graph = prepare_grid_register(graph_idx, parallel=not is_local, a=5.2e-6)
        parameters = {"omega_max": 1.58e7, "tau": 0.1}
        parameters.update({param_name: data["params_opt_det.params." + param_name] for param_name in param_names})
        H = prepare_drive(t_total, parameters, kind='lowpass')
        ahs_program = AnalogHamiltonianSimulation(hamiltonian=H, register=register)
        result = device.run(ahs_program, shots=n_shots).result()
        measured_bistrings = np.array([np.array(s.post_sequence) for s in result.measurements
                                       if np.all(s.pre_sequence)])

        bitstrings_dict[param_nr] = {"graph": graph_idx, "bitstrings": measured_bistrings.tolist()}
        with open(os.path.join(dir_path, "bitstrings.json"), "w") as f:
            json.dump(bitstrings_dict, f)
        print(f"Done with iteration {param_nr}.")
        print(bitstrings_dict)
        log_metric(metric_name="Iterations done", value=1, iteration_number=param_nr + 1)


def run_gridgraph_linear(hyperparams=None):
    """Runs linear schedules on the gridgraphs."""

    is_local = hyperparams is not None
    if not is_local:
        hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
        # input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]

        with open(hp_file, "r") as f:
            hyperparams = json.load(f)
        dir_path = os.environ['AMZN_BRAKET_JOB_RESULTS_DIR']
        device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])
        # device = LocalSimulator('braket_ahs')

    else:
        # input_dir = ""
        dir_path = ""
        device = LocalSimulator('braket_ahs')

    t_total = float(hyperparams["t_total"]) if "t_total" in hyperparams else 4e-6
    n_shots = int(hyperparams["n_shots"]) if "n_shots" in hyperparams else 60

    # if we choose the ,,optimized'' version
    if hyperparams["omega_max_type"] == "o":
        omega_max = 1.58e7
    # else we choose the naïve guess:
    else:
        omega_max = np.mean(get_parameter_bounds(0, optimize_rabi=True, scale=5.2/4.5)['omega_max'])
    parameters = {"tau": 0.1, "omega_max": omega_max, "delta_initial": 3e7, "delta_final": 6e7}
    print(parameters)
    bitstrings_dict = {}

    for gridgraph_index in range(11):
        register, graph = prepare_grid_register(gridgraph_index, parallel=not is_local, a=5.2e-6)
        H = prepare_drive(t_total, parameters, kind='lowpass')
        ahs_program = AnalogHamiltonianSimulation(hamiltonian=H, register=register)
        result = device.run(ahs_program, shots=n_shots).result()
        measured_bistrings = np.array([np.array(s.post_sequence) for s in result.measurements
                                       if np.all(s.pre_sequence)])
        bitstrings_dict[gridgraph_index] = measured_bistrings.tolist()
        with open(os.path.join(dir_path, f"bitstrings-linear-{hyperparams['omega_max_type']}.json"), "w") as f:
            json.dump(bitstrings_dict, f)

        print(f"Done with iteration {gridgraph_index + 1}.")
        print(bitstrings_dict)
        log_metric(metric_name="Iterations done", value=1 - gridgraph_index % 2, iteration_number=gridgraph_index + 1)
