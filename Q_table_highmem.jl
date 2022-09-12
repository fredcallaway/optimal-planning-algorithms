using ProgressMeter
using Distributed

function make_Q_table(data)
    println("Creating Q_table with high memory requirements")
    mkpath("$base_path/partial_Q_table")
    base_mdps = unique([d.t.m for d in data])
    M = map(Iterators.product(base_mdps, COSTS)) do (m, cost)
        mutate(m, cost=cost)
    end
    grouped_ids = eachcol(id.(M))
    function on_error(err)
        @error err
        if err isa InterruptException
            throw(err)
        end
        return err
    end
    # res = @showprogress map(grouped_ids) do ids
    res = @showprogress pmap(grouped_ids, on_error=on_error) do ids
        path = "$base_path/partial_Q_table/$(ids[1])"
        if isfile(path)
            println("Loading previously computed $path")
            return deserialize(path)
        end
        println("Computing $path")

        vs = map(load_V_nomem, ids)
        cost = vs[1].m.cost
        @assert all(v.m.cost == cost for v in vs)
        value_functions = Dict(v.m => v for v in vs)

        qs = map(data) do d
            V = value_functions[mutate(d.t.m, cost=cost)]
            @assert haskey(V.cache, V.hasher(V.m, d.b))
            Q(V, d.b)
        end
        serialize(path, qs)
        qs
    end
    all_qs = invert(res)

    @assert length(all_qs) == length(data)
    @assert length(all_qs[1]) == length(COSTS)
    map(data, all_qs) do d, dqs
        shash(d) => Dict(zip(COSTS, dqs))
    end |> Dict
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    @everywhere include("base.jl")
    all_trials = load_trials(EXPERIMENT)
    println("Loaded data for ", length(all_trials), " participants")
    data = all_trials |> values |> flatten |> get_data;
    serialize("$base_path/Q_table", make_Q_table(data))
    println("Wrote $base_path/Q_table")
end




