using Pkg
println("Installing packages...")
Pkg.add(readlines("deps.txt"))
Pkg.add(url="https://github.com/grero/StableHashes.jl")
Pkg.precompile()