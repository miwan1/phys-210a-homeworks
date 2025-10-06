# clt_random_walk_blocks.jl
using Random, Statistics
using StatsBase             # histogram binning
using Distributions         # Normal fit
using Plots                 # plotting
# Optional for a formal test:
import Pkg 
Pkg.add("HypothesisTests")
try
    using HypothesisTests
    HAS_HT = true
catch
    @warn "HypothesisTests.jl not found; skipping KS test."
    HAS_HT = false
end

# --- Parameters ---
Ntotal      = 2_000_000     # total number of steps in the long walk
step        = 1.0           # step size (± step with equal prob.)
block_size  = 400           # choose >> correlation time (for iid steps, any O(10^2)+ is fine)
seed        = 1234

Random.seed!(seed)

# --- Build a long random walk and compute block means of the increments ---
# In a simple 1D random walk, the *increments* are iid ±step.
# We'll block-average these increments to test CLT.
increments = @. ifelse(rand() < 0.5, step, -step)

nblocks = Ntotal ÷ block_size
if nblocks < 50
    error("Not enough blocks; increase Ntotal or reduce block_size.")
end

# Trim to an integer number of blocks and reshape into (block_size, nblocks)
trimmed = view(increments, 1:(nblocks*block_size))
X = reshape(trimmed, block_size, nblocks)

# Block means of increments (i.e., mean step per block)
block_means = vec(mean(X; dims=1))

# --- Theory vs fit ---
# CLT says: block_means ~ Normal(μ, σ^2) with μ = 0 and σ^2 = step^2 / block_size
theory_σ = step / sqrt(block_size)

# Fit a Normal by MLE to the block means
fitdist = fit(Normal, block_means)
μ̂, σ̂ = fitdist.μ, fitdist.σ

println("Number of blocks: $nblocks  (block_size = $block_size, Ntotal = $Ntotal)")
println("Sample mean      μ̂ = $(round(μ̂, digits=5))  (theory: 0)")
println("Sample stdev     σ̂ = $(round(σ̂, digits=5))  (theory: step/√B = $(round(theory_σ, digits=5)))")

# --- Histogram + PDF overlay ---
nbins = max(30, floor(Int, sqrt(nblocks)))  # a decent default
hist = fit(Histogram, block_means, nbins)
edges = hist.edges[1]
centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
counts = hist.weights

# Normalize to make it a probability histogram
binwidths = diff(edges)
pdf_hist = counts ./ (sum(counts) .* binwidths)

# Plot histogram and overlay fitted Normal & theoretical Normal
p1 = plot(centers, pdf_hist, seriestype=:sticks, label="Histogram (normalized)",
          xlabel="Block mean of increments", ylabel="Probability density",
          title="CLT check: block-mean distribution")

# Range for smooth PDFs
xmin, xmax = extrema(block_means)
xgrid = range(xmin - 3*σ̂, xmax + 3*σ̂, length=400)

plot!(xgrid, pdf.(fitdist, xgrid), lw=2, label="Gaussian fit  N(μ̂, σ̂²)")
plot!(xgrid, pdf.(Normal(0, theory_σ), xgrid), lw=2, ls=:dash, label="Theory  N(0, (step²/B))")

display(p1)

# --- Q–Q plot against the fitted Normal (should be ~ straight line) ---
sorted = sort(block_means)
prob   = ((1:length(sorted)) .- 0.5) ./ length(sorted)
q_theo = quantile.(Ref(fitdist), prob)

p2 = plot(q_theo, sorted, seriestype=:scatter, markersize=3,
          xlabel="Theoretical quantiles (fitted Normal)",
          ylabel="Empirical quantiles (block means)",
          title="Q–Q plot vs fitted Normal", label="data")
plot!(q_theo, q_theo, lw=2, label="y = x")
display(p2)

# --- Optional: KS test against the *theoretical* Normal N(0, step^2/B) ---
if HAS_HT
    ks = OneSampleKSTest(block_means, Normal(0, theory_σ))
    println("\nKolmogorov–Smirnov test vs N(0, step²/B):")
    println("  D = $(round(ks.D, digits=4)),  p-value = $(round(pvalue(ks), digits=4))")
    println("  (Failing to reject at common α indicates good agreement.)")
end
