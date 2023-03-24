using OpenQuantumTools, OrdinaryDiffEq, Plots, LinearAlgebra, Arpack, Interpolations
using BayesianOptimization, GaussianProcesses, Distributions
using NPZ


# redefine diagonalization for sparse Hams
import OpenQuantumTools: haml_eigs

function haml_eigs(H::SparseHamiltonian, t, lvl)
        hmat = H(t)
        w, v = eigs(hmat, nev = lvl, which=:SR, tol=1e-5, maxiter=100)
        real.(w), v
end


function sigz(j)
    """Sigma-z operator in the reduced Hilbert space of conserved total angular momentum."""
    return complex.(Diagonal(collect(-j:j)))
end

function sigx(j)
    """Sigma-x operator in the reduced Hilbert space of conserved total angular momentum."""
    d = Int(2 * j + 1)
    arr = zeros(d, d)

    for x in 1:d
        m = x - j - 1
        if x + 1 <= d
            arr[x, x+1] += sqrt(j * (j + 1) - m * (m + 1))
        end
        if x - 1 > 0
            arr[x, x-1] += sqrt(j * (j + 1) - m * (m - 1))
        end
    end
    return complex.(arr / 2)
end

function pspin(n, c=0, p=3)
    """p-spin hamiltonian in the J=max subspace."""
    dim1, dim2 = round(Int, n * c), n - round(Int, n * c)
    return complex.(- n * ((2 / n) * (
                        kron(sigz(dim1 / 2), I(dim2 + 1))
                            +
                             kron(I(dim1 + 1), sigz(dim2 / 2))
                        )
                  ) ^ p)
end

function h_init(n, c)
    """Initial solution hamiltonian in the J=max subspace."""
    dim1, dim2 = round(Int, n * c), n - round(Int, n * c)
    return complex.(-2 * (kron(sigz(dim1 / 2), I(dim2 + 1)) - kron(I(dim1 + 1), sigz(dim2 / 2))))
end

function v_tf(n, c=0)
    """Transverse field hamiltonian in the J=max subspace"""
    dim1, dim2 = round(Int, n * c), n - round(Int, n * c)
    return complex.(-2 * (kron(sigx(dim1 / 2), I(dim2 + 1)) + kron(I(dim1 + 1), sigx(dim2 / 2))))
end


tts(τ, pg; pd=0.99) = τ * log(1 - 0.99) / log(1 - pg)


function build_ara(n, c, gamma; s=(x)->x, λ=(x)->x)
    """Builds the Adiabatic Reverse Annealing (ARA) Hamiltonian."""
    psp = (pspin(n, c, 3))
    hin = (h_init(n, c))
    vtf = (gamma * v_tf(n, c))

    H = SparseHamiltonian([(x)->s(x), (x)->(1-s(x))*(1-λ(x)), (x)->(1 - s(x))*λ(x)],
     [psp, hin, vtf], unit=:ħ)
    return H
end


function build_bath_coupling(η, fc, T, n, c)
    """Builds an Ohmic bath and a coupling operator corresponding to collective dephasing --
    basically just a σ_z operator."""
    bath = Ohmic(η, fc, T)

    # we recycle the pspin function to create the σ_z operator:
    coupling = CustomCouplings([(s)->-pspin(n, c, 1)], unit=:ħ)

    return bath, coupling
end


function build_annealing(H, bath, coupling)
    """Builds the annealing object, starting from the ground state of the Hamiltonian."""
    u0 = vec(haml_eigs(H, 0, 1)[2])

    annealing = Annealing(H, u0; coupling=coupling, bath=bath)
    return annealing
end


function realpath(points)
    """Returns a function which interpolates between points."""
    points = append!([0.0], points, [1.0])
    xpoints = 0.0: 1/(length(points) - 1) :1.0
    return Interpolations.linear_interpolation(xpoints, points)
end


function prepare_annealing(η,  T, n, c, gamma; s=(x)->x, λ=(x)->x, fc=8 * π)
    H = build_ara(n, c, gamma, s=s, λ=λ)
    bath, coupling = build_bath_coupling(η, fc, T, n, c)
    return build_annealing(H, bath, coupling)
end

function prepare_independent_annealing(η,  T, n, c, gamma; u1=(x)->x, u2=(x)->x, fc=8 * π)
    psp = (pspin(n, c, 3))
    vtf = (gamma * v_tf(n, c))

    H = SparseHamiltonian([x->1-u1(x), x->u2(x)], [vtf, psp], unit=:ħ)
    bath, coupling = build_bath_coupling(η, fc, T, n, c)
    return build_annealing(H, bath, coupling)
end


function get_objective(η,  T, n, c, gamma, tf; kind="r", fc=8*π, n_saved=100, eq="unitary", lvl=10)
    function _objective(parameters)
        if kind == "r"
            lp = length(parameters)
            s_params = parameters[1:lp÷2]
            λ_params = parameters[lp÷2 + 1 : lp]
#             print(s_params, λ_params)
            s = realpath(s_params)
            λ = realpath(λ_params)
            annealing = prepare_annealing(η,  T, n, c, gamma, s=s, λ=λ, fc=fc)
        elseif startswith(kind, "q")
            s = realpath(parameters)
            λ = (x) -> 1.0
            c = 1.0
            annealing = prepare_annealing(η,  T, n, c, gamma, s=s, λ=λ, fc=fc)
        elseif startswith(kind, "i")
            lp = length(parameters)
            u1_params = parameters[1:lp÷2]
            u2_params = parameters[lp÷2 + 1 : lp]
            u1 = realpath(u1_params)
            u2 = realpath(u2_params)
            c = 1.0
            annealing = prepare_independent_annealing(η,  T, n, c, gamma, u1=u1, u2=u2, fc=fc)
        else
            error("Kind $kind not allowed.")
        end
        tlist = 0:tf/(n_saved - 1):tf
        if eq == "unitary"
            sol = solve_schrodinger(annealing, tf,
                                alg=Tsit5(), reltol=1e-6, saveat=0:tf/(n_saved - 1):tf)
            return abs(sol[n_saved][length(sol[n_saved])])^2
        elseif eq == "ame"
            sol = solve_ame(annealing, tf; alg=Tsit5(), ω_hint=range(-8, 8, length=200), reltol=1e-6,
                            lvl=lvl, saveat=tlist)
            return abs(sol[n_saved][size(sol[n_saved])...])
#         println("Length $(length(sol.u))")
        else
            error("eq $eq not implemented.")
        end

    end
    return _objective
end

get_linear_params(n_params::Int) = Vector((0.:1/(n_params + 1):1)[2:n_params+1])

function bayesian_optimize(f, n_params; n_initialpoints=9, n_iterations=50, kind="r", span=1)
    model = ElasticGPE(n_params, kernel = Mat52Iso(1., -1.), logNoise=-5)
    if (kind == "r") | (kind == "i")
        lp = get_linear_params(n_params÷2)
        linear_parameters = append!(lp, lp)
        low_bounds = [max(v - span / (length(lp) + 1), 0) for v in linear_parameters]
        up_bounds = [min(v + span / (length(lp) + 1), 1) for v in linear_parameters]
    elseif kind == "q"
        linear_parameters = get_linear_params(n_params)
        low_bounds = [max(v - span / (length(linear_parameters) + 1), 0) for v in linear_parameters]
        up_bounds = [min(v + span / (length(linear_parameters) + 1), 1) for v in linear_parameters]
    else
        error("Kind $kind not allowed.")
    end
    initial_points = append!([linear_parameters], [rand(n_params) for _=1:n_initialpoints])
    initial_values = f.(initial_points)

    append!(model, hcat(initial_points...), initial_values)

#     low_bounds = [0. for _=1:n_params]
#     up_bounds = [1. for _=1:n_params]
    # for now we keep this here..
    modeloptimizer = MAPGPOptimizer(every = 10,
                                noisebounds = [-7, -3],       # bounds of the logNoise
                                kernbounds = [[-5, -5], [5, 5]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                maxeval = 100)
    opt = BOpt(f,
               model,
               UpperConfidenceBound(scaling=BrochuBetaScaling(.1), βt=2.),                # type of acquisition
               modeloptimizer,
               low_bounds, up_bounds,                     # lowerbounds, upperbounds
               repetitions = 1,                          # evaluate the function for each input 5 times
               maxiterations = n_iterations,                      # evaluate at 100 input positions
               sense = Max,
               initializer_iterations = 0,
               acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl) end
               verbosity = Progress)
    return boptimize!(opt), model
end


#
matricize(vec) = mapreduce(permutedims, vcat, vec)

function run_experiments()
    ns = [5, 10, 15, 20, 25, 30]
    ηs = [0., 1e-6, 1e-4]

    c = 0.8
    gamma = 5

    # bath and coupling parameters

    fc = 8 * π
    T = 12


    n_pars = 4
    tf = 20

    res = []
    for n in ns
        dim_r = Int(round(n * c + 1) * round(n * (1-c) + 1))
        dim_q = Int(round(n + 1))
        lvl_r = min(dim_r - 2, 50)
        lvl_q = min(dim_q - 2, 50)
        tmp = []
        for η in ηs
            if η == 0.0
                f_r = get_objective(η,  T, n, c, gamma, tf; kind="r", eq="unitary")
                f_q = get_objective(η,  T, n, c, gamma, tf; kind="q", eq="unitary")
            else
                f_r = get_objective(η,  T, n, c, gamma, tf; kind="r", eq="ame", lvl=lvl_r)
                f_q = get_objective(η,  T, n, c, gamma, tf; kind="q", eq="ame", lvl=lvl_q)
            end
            println("Q Ann")
            prob_q = @time bayesian_optimize(f_q, n_pars; span=2)[1].observed_optimum
            println("R Ann")
            prob_r = @time bayesian_optimize(f_r, n_pars; span=1)[1].observed_optimum
            tmp = push!(tmp, [prob_q, prob_r])
            println("n = $n; η = $η; DONE")
            flush(stdout)
        end
        res = push!(res, matricize(tmp))
    end
    return reshape(matricize(res), length(ns), length(ηs), 2)
end

function run_experiments_independent()
    ns = [5, 10, 15, 20, 25, 30]
    ηs = [0., 1e-6, 1e-4]

    c = 0.8
    gamma = 5

    # bath and coupling parameters

    fc = 8 * π
    T = 12


    n_pars = 4
    tf = 20

    res = []
    for n in ns
        dim_i = Int(round(n + 1))
        lvl_i = min(dim_i - 2, 30)
        tmp = []
        for η in ηs
            if η == 0.0
                f_i = get_objective(η,  T, n, c, gamma, tf; kind="i", eq="unitary")
            else
                f_i = get_objective(η,  T, n, c, gamma, tf; kind="i", eq="ame", lvl=lvl_i)
            end
            println("Q Ann (indep)")
            prob_i = @time bayesian_optimize(f_i, n_pars; span=2)[1].observed_optimum
            tmp = push!(tmp, prob_i)
            println("n = $n; η = $η; DONE")
            flush(stdout)
        end
        res = push!(res, tmp)
    end
    return matricize(res)
end


# 30 and 50 levels were used!!!

import Logging
Logging.disable_logging(Logging.Warn);
# @sync begin @parallel for i in processor_range
#     push!(res, run_experiments())
#     end
# end
res = run_experiments_independent()

# change into a proper multidim tensor
# output = reshape(reduce(hcat, res), length(processor_range), size(res[1])...)
# writing data such that it is straightforwardly readable by numpy
npzwrite("data/pspin/ame-data-test-$(ARGS[1]).npz", Float64.(res))

