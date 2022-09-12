using StatsFuns: softplus, invsoftplus, logit
using DataStructures: OrderedDict

Space = OrderedDict{Symbol,Any}

function bounds(space::Space)
    x = (Float64[], Float64[], Float64[], Float64[])
    for spec in values(space)
        if length(spec) == 4
            for i in 1:4
                push!(x[i], spec[i])
            end
        end
    end
    x
end

function combinations(space::Space)
    specs = filter(collect(values(space))) do spec
        spec isa Vector
    end
    Iterators.product(specs...)
end

struct Box
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function squash!(bb::Box, x)
    for i in eachindex(x)
        lo = bb.lower[i]; hi = bb.upper[i]
        if lo == -Inf && hi == Inf
            continue
        elseif lo == -Inf
            x[i] = hi - softplus(-x[i])
        elseif hi == Inf
            x[i] = lo + softplus(x[i])
        else
            x[i] = lo + (hi - lo) * logistic(x[i])
        end
    end
    #@assert all(bb.lower .<= x .<= bb.upper)
    x
end

function unsquash!(bb::Box, y)
    #@assert all(bb.lower .<= y .<= bb.upper)
    for i in eachindex(y)
        lo = bb.lower[i]; hi = bb.upper[i]
        if lo == -Inf && hi == Inf
            continue
        elseif lo == -Inf
            # y = hi - softplus(-x)
            # softplus(-x) = hi - y
            y[i] = -invsoftplus(hi - y[i])
        elseif hi == Inf
            #y = lo + softplus(x)
            y[i] = invsoftplus(y[i] - lo)
        else
            # y = lo + (hi - lo) * logistic(x)
            # y - lo = (hi - lo) * logistic(x)
            # (y - lo) / (hi - lo) = logistic(x)
            y[i] = logit((y[i] - lo) / (hi - lo))
        end
    end
    y
end
