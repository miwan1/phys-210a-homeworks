# For plotting:
import Pkg; Pkg.activate(temp=true); Pkg.add("Plots"); Pkg.add("StatsBase"); Pkg.add("Distributions")

# random_walk_msd.jl
using Random, Statistics, StatsBase, Distributions, Plots



"""
    msd_random_walk(nsteps=2000, ntrials=10000; step=1.0, seed=42)

Simulate `ntrials` independent 1D random walks of `nsteps` steps,
each step being ±`step` with equal probability. Returns the vector of
mean-square displacements at each time step, and the fitted slope.
"""
function msd_random_walk(nsteps::Int=2000, ntrials::Int=10000; step=1.0, seed=42)
    Random.seed!(seed)

    # Memory-efficient ensemble average over trials
    msd_acc = zeros(Float64, nsteps)

    for _ in 1:ntrials
        x = 0.0
        @inbounds for t in 1:nsteps
            x += ifelse(rand() < 0.5, step, -step)
            msd_acc[t] += x^2
        end
    end

    msd = msd_acc ./ ntrials
    t = collect(1:nsteps)

    # Fit ⟨x^2⟩ ≈ a * t through the origin (theory: a ≈ step^2)
    slope = sum(t .* msd) / sum(t .^ 2)

    # Optional: also estimate intercept to check it's ~0
    intercept = mean(msd .- slope .* t)

    return t, msd, slope, intercept
end



"""
    clt_check_blocks(; longwalk_steps=2_000_000, block_size=400, step=1.0,
                      seed=1234, outfile="clt_distribution.png")

Generate one very long 1D random-walk *increment* sequence (±step), split it into
non-overlapping blocks (size >> correlation time), compute the mean increment in
each block, and check that their distribution is Gaussian.

Saves a PNG histogram with fitted and theoretical Normal overlays.
Returns (mu_hat, sigma_hat, sigma_theory, nblocks).
"""
function clt_check_blocks(; longwalk_steps::Int=2_000_000, block_size::Int=400,
                          step::Float64=1.0, seed::Int=1234,
                          outfile::AbstractString="clt_distribution.png")
    @assert block_size > 0 "block_size must be positive"
    Random.seed!(seed)

    # iid increments: ±step with prob 1/2
    increments = [rand() < 0.5 ? step : -step for _ in 1:longwalk_steps]

    # Split into blocks and compute block means (mean displacement per step in block)
    nblocks = longwalk_steps ÷ block_size
    @assert nblocks ≥ 50 "Not enough blocks; increase longwalk_steps or reduce block_size."

    trimmed = @view increments[1:(nblocks * block_size)]
    X = reshape(trimmed, block_size, nblocks)
    block_means = vec(mean(X; dims=1))

    # Sample stats and theory
    mu_hat = mean(block_means)
    sigma_hat = std(block_means)
    sigma_theory = step / sqrt(block_size)  # CLT: Var(mean) = step^2 / block_size

    # Histogram + Gaussian overlays (save as PNG)
    nbins = max(30, floor(Int, sqrt(nblocks)))
    p = histogram(block_means, nbins=nbins, norm=true, label="Histogram",
                  xlabel="Block mean displacement", ylabel="Probability density")
    xgrid = range(minimum(block_means), maximum(block_means), length=400)
    plot!(xgrid, pdf.(Normal(mu_hat, sigma_hat), xgrid), lw=2,
          label="Gaussian fit")
    plot!(xgrid, pdf.(Normal(0, sigma_theory), xgrid), lw=2, ls=:dash,
          label="Theory")

    savefig(p, outfile)
    println("CLT: μ̂=$(round(mu_hat, digits=5)) (theory 0),  σ̂=$(round(sigma_hat, digits=5)) ",
            "(theory $(round(sigma_theory, digits=5))).  Saved '$outfile'.")

    return mu_hat, sigma_hat, sigma_theory, nblocks
end



"""
    clt_check_blocks_correlated(; longwalk_steps=2_000_000, block_size=400,
                                 rho=0.9, step=1.0, seed=1234,
                                 outfile="clt_correlated_distribution.png")

Generate one long sequence of *correlated* increments via an AR(1)/OU process:
    s_{n+1} = rho * s_n + η_n,  η_n ~ N(0, 1 - rho^2)
scaled so that std(s) = `step`. Split into non-overlapping blocks, take the
mean in each block, and check CLT by plotting a histogram with Gaussian overlays.

Returns (mu_hat, sigma_hat, sigma_theory, nblocks, tau_int). Saves a PNG to `outfile`.
"""
function clt_check_blocks_correlated(; longwalk_steps::Int=2_000_000,
                                      block_size::Int=400, rho::Float64=0.9,
                                      step::Float64=1.0, seed::Int=1234,
                                      outfile::AbstractString="clt_correlated_distribution.png")

    @assert -1.0 < rho < 1.0 "rho must satisfy |rho| < 1"
    @assert block_size > 0 "block_size must be positive"

    Random.seed!(seed)

    # AR(1) with stationary var = 1; then scale by `step`
    sigma_eta = sqrt(1 - rho^2)
    s = Vector{Float64}(undef, longwalk_steps)
    s[1] = randn() * step                  # start from stationary distribution (≈)
    for n in 2:longwalk_steps
        s[n] = rho * s[n-1] + randn() * sigma_eta * step
    end

    # Block means (use non-overlapping blocks)
    nblocks = longwalk_steps ÷ block_size
    @assert nblocks ≥ 50 "Not enough blocks; increase longwalk_steps or reduce block_size."

    trimmed = @view s[1:(nblocks * block_size)]
    X = reshape(trimmed, block_size, nblocks)
    block_means = vec(mean(X; dims=1))

    # Empirical stats
    mu_hat = mean(block_means)
    sigma_hat = std(block_means)

    # Theory for large blocks: Var(mean) ≈ step^2 * τ_int / B
    tau_int = (1 + rho) / (1 - rho)              # integrated autocorrelation time
    sigma_theory = step * sqrt(tau_int / block_size)

    # Histogram + overlays (PNG)
    nbins = max(30, floor(Int, sqrt(nblocks)))
    p = histogram(block_means, nbins=nbins, norm=true, label="Histogram",
                  xlabel="Correlated Block-mean displacement",
                  ylabel="Probability density")
    xgrid = range(minimum(block_means), maximum(block_means), length=400)
    plot!(xgrid, pdf.(Normal(mu_hat, sigma_hat), xgrid), lw=2,
          label="Gaussian fit")
    plot!(xgrid, pdf.(Normal(0, sigma_theory), xgrid), lw=2, ls=:dash,
          label="Theory")

    savefig(p, outfile)
    println("CLT (correlated):  μ̂=$(round(mu_hat, digits=5)) (theory 0),  σ̂=$(round(sigma_hat, digits=5))",
            "  vs  σ_theory=$(round(sigma_theory, digits=5));  τ_int=$(round(tau_int, digits=3)); ",
            "blocks=$nblocks.  Saved '$outfile'.")

    return mu_hat, sigma_hat, sigma_theory, nblocks, tau_int
end
#3(a)
# ---- Run it and visualize ----
nsteps   = 1000
ntrials  = 10000
step     = 1.0

t, msd, slope, intercept = msd_random_walk(nsteps, ntrials; step=step, seed=123)

println("Fitted MSD ~ a * t with")
println("  a (slope)      = $(round(slope, digits=4))   (theory: step^2 = $(step^2))")
println("  intercept      = $(round(intercept, digits=4)) (should be ~ 0)")
println("  ratio a/step^2 = $(round(slope / step^2, digits=4))")

# Plot MSD and the best-fit line
p = plot(t, msd, label="Mean square displacement", xlabel="time steps (t)",
         ylabel="⟨ΔX²⟩", lw=2)
plot!(t, slope .* t, label="Linear fit: $(round(slope, digits=3)) · t", lw=2, ls=:dash)

display(p)
# savefig(p, "/Users/mingzewanmba/Downloads/PHYS 210A/homeworks/hw1/fig/walk1D_10000trials.png")

#3(b)(1)
block_size = 100
step = 1.0
longwalk_steps = 2000000
outfile = "/Users/mingzewanmba/Downloads/PHYS 210A/homeworks/hw1/fig/clt.png"
mu, sig, sig_theory, nblocks = clt_check_blocks(block_size=block_size, step=step, longwalk_steps=longwalk_steps, outfile=outfile)

#3(b)(2)
mu, sig, sig_th, nb, tau = clt_check_blocks_correlated(
    rho=0.9, block_size=100, step=1.0,
    longwalk_steps=2000000,
    outfile="/Users/mingzewanmba/Downloads/PHYS 210A/homeworks/hw1/fig/clt_correlated_distribution.png")