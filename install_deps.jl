using Pkg
println("Installing packages...")
Pkg.add(readlines("deps.txt"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/grero/StableHashes.jl"))
Pkg.precompile()