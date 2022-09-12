struct Likelihood
    data::Vector{Datum}
end

Likelihood(trials::Vector{Trial}) = Likelihood(get_data(trials))
n_action(L) = length(L.data[1].b) + 1
n_datum(L) = length(L.data)

@memoize memo_map(f::Function, L::Likelihood) = map(f, L.data)

function logp(model::AbstractModel, trials::Vector{Trial})
    L = Likelihood(get_data(trials))
    logp(L, model)
end

function logp(L::Likelihood, model::M)::T where M <: AbstractModel{T} where T <: Real
    φ = memo_map(L) do d
        features(M, d)
    end

    tmp = zeros(T, n_action(L))
    total = zero(T)
    for i in eachindex(L.data)
        a = L.data[i].c + 1
        p = action_dist!(tmp, model, φ[i])
        # if !(sum(p) ≈ 1)
        #     @error "bad probability vector" p sum(p)
        # end
        # @assert sum(p) ≈ 1
        @assert isfinite(p[a])
        total += log(p[a])
    end
    total
end

function random_restarts(loss, hard_lower, soft_lower, soft_upper, hard_upper, n_restart; 
                         #algorithm=LBFGS(), 
                         algorithm=LBFGS(),
                         iterations=100, g_tol=1e-5,
                         max_err=30, max_timeout=50, max_finite=30, id="null")
    #algorithms = [
    #    Fminbox(LBFGS()),
    #    Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())),
    #] |> Iterators.cycle |> Iterators.Stateful
    n_err = 0
    n_time = 0
    n_finite = 0
    box = Box(hard_lower, hard_upper)
    function wrap_loss(x)
        squashed = squash!(box, copy(x))
        loss(squashed) + 1e-8 * sum(x .^ 2)  # tiny regularization stabilizes optimization
    end

    function do_opt(x0)
        if !isfinite(loss(x0))  # hopeless!
            n_finite += 1
            @debug "nonfinite loss" n_finite
            if n_finite > max_finite
                @error "$id: Too many non-finite losses while optimizing"
                error("Optimization non-finite")
            end
            return missing
        end
        try
            # >~30s indicates that the optimizer is stuck, which means it's not likely to find a good minimum anyway
            res = optimize(wrap_loss, unsquash!(box, copy(x0)), algorithm,
                Optim.Options(;g_tol, iterations); autodiff=:forward)
            if !(res.f_converged || res.g_converged) && res.iterations > iterations
                n_time += 1
                @debug "timeout" n_time
                if n_time > max_timeout
                    @error "$id: Too many timeouts while optimizing"
                    error("Optimization timeouts")
                end
                return missing
            elseif !isfinite(wrap_loss(res.minimizer))
                @error "Nonfinite final loss!" wrap_loss(res.minimizer)
                return missing
            else
                squash!(box, res.minimizer)
                return res
            end
        catch err
            err isa InterruptException && rethrow(err)
            n_err += 1
            #@warn "error" err n_err
            if n_err > max_err
                @error "$id: Too many errors while optimizing"
                rethrow(err)
            end
            return missing
        end
    end

    x0s = SobolSeq(soft_lower, soft_upper)
    results = Any[]
    while length(results) < n_restart
        res = do_opt(next!(x0s))
        if !ismissing(res)
            push!(results, res)
        end
    end
    if n_err > max_err/2 || n_time > max_timeout/2 || n_finite > max_finite/2
        @warn "$id: Difficulty optimizing" n_err n_time n_finite
    end
    losses = getfield.(results, :minimum)
    very_good = minimum(losses) * 1.05
    n_good = sum(losses .< very_good)
    #if n_good < 5
    #    best_losses = partialsort(losses, 1:5)
    #    @warn "$id: Only $n_good random restarts produced a very good minimum" best_losses
    #end
    partialsort(results, 1; by=o->o.minimum)  # best result
end

function Distributions.fit(::Type{M}, trials::Vector{Trial}; 
            method=:bfgs, n_restart=100, opt_kws...) where M <: AbstractModel
    space = default_space(M)
    hard_lower, soft_lower, soft_upper, hard_upper = all_bounds = bounds(space)
    L = Likelihood(trials)
    empty!(memoize_cache(memo_map))
    
    if isempty(hard_lower)  # no free parameters
        model = create_model(M, hard_lower, (), space)
        return model, -logp(L, model)
    end

    function make_loss(z)
        x -> begin
            model = create_model(M, x, z, space)
            # L1 = sum(abs.(x) ./ space_size)
            -logp(L, model) #+ 10 * L1
        end
    end

    results, elapsed = @timed map(combinations(space)) do z
        loss = make_loss(z)

        opt = begin
            if method == :samin
                x0 = soft_lower .+ rand(length(lower)) .* (soft_upper .- soft_lower)
                optimize(loss, hard_lower, hard_upper, x0, SAMIN(verbosity=0), Optim.Options(iterations=10^6))
            elseif method == :bfgs
                t = trials[1]
                id = "$(name(M))-$(t.wid)-$(t.i)"
                random_restarts(loss, all_bounds..., n_restart; id, opt_kws...)
            end
        end
        ismissing(opt) && return missing
        model = create_model(M, opt.minimizer, z, space)
        if !isfinite(logp(L, model))
            id = "$(name(M))-$(t.wid)-$(t.i)"
            @error "NONFINITE LOSS" model id
            @show model
        end
        model, -logp(L, model)
    end |> skipmissing |> collect 
    if isempty(results)
        @error("Could not fit $M to $(trials[1].wid)")
        error("Fitting error")
    end
    # @info "Fitting complete" n_call elapsed M
    models, losses = invert(results)
    i = argmin(losses)
    models[i], -logp(L, models[i])
end
