#=
julia_playground:
- Julia version: 
- Author: q521093
- Date: 2022-10-27
=#

using Bloqade, BloqadeMIS, GenericTensorNetworks, DelimitedFiles, CUDA, Statistics


function random_square_udg(n, dropout; return_atoms=false, dist=7.5)
    atoms = generate_sites(SquareLattice(), n, n; scale=4.5) |> random_dropout(dropout)
    if return_atoms
        return BloqadeMIS.unit_disk_graph(atoms, dist), atoms
    else
        return BloqadeMIS.unit_disk_graph(atoms, dist)
    end
end

function get_hardness_param(graph; cuda=false, return_size=false)
    problem = IndependentSet(graph; optimizer=TreeSA());
    n_mis_m1, n_mis = GenericTensorNetworks.solve(problem, CountingMax(2), usecuda=cuda)[].coeffs
    size_mis = GenericTensorNetworks.solve(problem, SizeMax(), usecuda=cuda)[].n
    if return_size
        return (n_mis_m1 / (size_mis * n_mis), size_mis)
    else
        return n_mis_m1 / (size_mis * n_mis)
    end
end

function generate_hard_graphs(sizes, hardness_percentile, n_attempts; dropout=0.3, cuda=false, dist=7.5)
    graphs = []
    hps = []
    mis_list = []
    for nv in sizes
        tmp_graphs = []
        tmp_hps = []
        tmp_mis_list = []
        for n in 1:n_attempts
            graph, atoms = random_square_udg(nv, dropout; return_atoms=true, dist=dist)
            println("Generated graph $n of size $(nv^2)")
            @time hp, mis_size = get_hardness_param(graph; cuda=cuda, return_size=true)
            push!(tmp_graphs, atoms)
#             print(tmp_graphs)
            push!(tmp_hps, hp)
            push!(tmp_mis_list, mis_size)
        end
        graphs = vcat(graphs, tmp_graphs)
        hps = vcat(hps, tmp_hps)
        mis_list = vcat(mis_list, tmp_mis_list)
        println("Finished generating graphs of size $(nv^2)")
    end
#     graphs = reshape(graphs, length(sizes), n_attempts)
#     println(graphs)

#     hps = reshape(hps, length(sizes), n_attempts)

    cutoff = Int(round(hardness_percentile * n_attempts * size(sizes)[1]))
    indices = sortperm(hps, rev=true)[1:cutoff]
    return ([graphs[i] for i in indices], [[mis_list[i], hps[i]] for i in indices])
end

function probe_dropouts(n_nodes, n_reps, dropouts; dist=7.5)
    hps = []
    for dropout in dropouts
        tmp = []
        for _=1:n_reps
            graph = random_square_udg(n_nodes, dropout; return_atoms=false, dist=dist)
            @time hp = get_hardness_param(graph)
            push!(tmp, hp)
        end
        push!(hps, mean(tmp))
    end
    return hps
end
minsize, maxsize, hp, n_att = ARGS

minsize = parse(Int, minsize)
maxsize = parse(Int, maxsize)
hp = parse(Float64, hp)
n_att = parse(Int, n_att)
dist = 4.5 * 1.5

gs, hp_mis_list = generate_hard_graphs(minsize:maxsize, hp, n_att; dropout=0.3, dist=dist)
# print(gs)
println("data/mis/graphs/hard_graphs-$minsize-$maxsize-$hp-$dist.txt")
writedlm("data/mis/graphs/hard_graphs-$minsize-$maxsize-$hp-$dist.txt", gs)
writedlm("data/mis/graphs/hp_mis-$minsize-$maxsize-$hp-$dist.txt", hp_mis_list)


# n_nodes = parse(Int, n_nodes)
# n_reps = parse(Int, n_reps)
# d_min = parse(Float64, d_min)
# d_max = parse(Float64, d_max)
# nds = parse(Int, nds)
# dist = parse(Float64, ARGS[1])
# #
# n_nodes = 16
# n_reps = 20
# d_min = 0.01
# d_max = 0.5
# nds = 6
#
#
# dropouts = range(d_min, d_max, length=nds)
# hps = probe_dropouts(n_nodes, n_reps, dropouts; dist=dist)
# writedlm("data/dropout_stats-$n_nodes-$d_min-$d_max-$dist.txt", hps)

# dist 8, dropout 0.3