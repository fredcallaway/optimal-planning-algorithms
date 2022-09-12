using Distributed
using ProgressMeter
@everywhere include("utils.jl")
@everywhere include("mdp.jl")
@everywhere include("data.jl")

COSTS = [0:0.05:4; 100]

mkpath("mdps/withcost")
mkpath("mdps/V")


function sbatch_script(n; minutes=20, memory=5000)
    """
    #!/usr/bin/env bash
    #SBATCH --job-name=solve
    #SBATCH --output=out/%A_%a
    #SBATCH --array=1-$n
    #SBATCH --time=$minutes
    #SBATCH --mem-per-cpu=$memory
    #SBATCH --cpus-per-task=1
    #SBATCH --mail-type=end
    #SBATCH --mail-user=flc2@princeton.edu

    module load julia/1.4.1
    julia solve.jl \$SLURM_ARRAY_TASK_ID
    """
end

function write_mdps(ids)
    base_mdps = map(ids) do i
        deserialize("mdps/base/$i")c
    end
    all_mdps = [mutate(m, cost=c) for m in base_mdps, c in COSTS]
    
    files = String[]
    for m in base_mdps, c in COSTS
        mc = mutate(m, cost=c)
        f = "mdps/withcost/$(id(mc))"
        serialize(f, mc)
        push!(files, f)
    end
    unsolved = filter(files) do f
        !isfile(replace(f, "withcost" => "V"))
    end

    unsolved = [string(split(f, "/")[end]) for f in unsolved]
    serialize("tmp/unsolved", unsolved)
    unsolved
end

write_mdps() = write_mdps(readdir("mdps/base"))

@everywhere function solve_mdp(i::String)
    m = deserialize("mdps/withcost/$i")
    if isfile("mdps/V/$i")
        println("MDP $i has already been solved.")
        return
    end

    V = ValueFunction(m)
    println("Begin solving MDP $i:  cost = $(m.cost),  hasher = $(V.hasher)"); flush(stdout)
    @time v = V(initial_belief(m))
    println("Value of initial state is ", v)
    serialize("mdps/V/$i", V)
    V = nothing
    GC.gc()
end

@everywhere do_job(id::String) = solve_mdp(id)
@everywhere do_job(idx::Int) = solve_mdp(deserialize("tmp/unsolved")[idx])
do_job(jobs) = pmap(solve_mdp, deserialize("tmp/unsolved")[jobs])

function solve_all()
    todo = write_mdps()
    println("Solving $(length(todo)) mdps.")
    do_job(eachindex(todo))
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    if ARGS[1] == "setup"
        todo = write_mdps()
        open("solve.sbatch", "w") do f
            write("solve.sbatch", sbatch_script(length(todo)))
        end

        println(length(todo), " mdps to solve with solve.sbatch")
    else  # solve an MDP (or several)
        if ARGS[1] == "all"
            solve_all()
        elseif startswith(ARGS[1], "exp")
            include("conf.jl")
            mdps = let
                all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
                flat_trials = flatten(values(all_trials));
                unique(getfield.(flat_trials, :m))
            end
            all_ids = map(Iterators.product(mdps, COSTS)) do (m, cost)
                id(mutate(m, cost=cost))
            end
            @showprogress pmap(solve_mdp, all_ids)
        else
            do_job(eval(Meta.parse(ARGS[1])))
        end
    end

end
