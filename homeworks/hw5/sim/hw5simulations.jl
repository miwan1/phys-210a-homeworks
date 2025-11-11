# For plotting:
import Pkg; 
Pkg.activate(temp=true); 
Pkg.add("Plots"); 
Pkg.add("StatsBase"); 
Pkg.add("Distributions"); 
Pkg.add("Random");


using Random
using Statistics
using Plots

# --- utilities ---

# angle difference wrapped to (-π, π]
@inline function angdiff(a, b)
    d = a - b
    # wrap
    d = mod(d + π, 2π) - π
    return d
end

# reflect a point off a circle of radius R centered at origin
# incoming position p has radius r > R; we mirror across the circle
@inline function reflect_off_circle(x, y, R)
    r = hypot(x, y)
    # mirrored radius = 2R - r
    scale = (2R - r) / r
    return x * scale, y * scale
end

# one Brownian step in 2D
@inline function brownian_step(x, y, D, dt)
    σ = sqrt(2D*dt)
    return x + σ*randn(), y + σ*randn()
end

# simulate winding for "pointlike flagpole, bounded field"
function simulate_winding_pointlike(; 
        D=1.0, dt=1e-3, T=1.0, 
        R=1.0,               # outer radius of the field
        x0=0.5, y0=0.0       # starting point (inside R)
    )
    steps = Int(round(T/dt))
    x, y = x0, y0
    θ = 0.0
    prev_ang = atan(y, x)
    ϵ = 1e-6                 # avoid exact zero radius

    for _ in 1:steps
        # Brownian proposal
        xnew, ynew = brownian_step(x, y, D, dt)

        # reflect off outer boundary
        rnew = hypot(xnew, ynew)
        if rnew > R
            xnew, ynew = reflect_off_circle(xnew, ynew, R)
            rnew = hypot(xnew, ynew)
        end

        # avoid division by zero right at the origin
        if rnew < ϵ
            # nudge outward slightly
            xnew += ϵ
        end

        # update winding angle via angle differences
        ang = atan(ynew, xnew)
        θ += angdiff(ang, prev_ang)
        prev_ang = ang

        x, y = xnew, ynew
    end

    return θ
end

# simulate winding for "finite radius a flagpole, bounded field"
function simulate_winding_annulus(; 
        D=1.0, dt=1e-3, T=1.0, 
        R=1.0, a=0.2,           # outer and inner radii
        x0=0.6, y0=0.0          # start in the annulus
    )
    steps = Int(round(T/dt))
    x, y = x0, y0
    θ = 0.0
    prev_ang = atan(y, x)

    for _ in 1:steps
        xnew, ynew = brownian_step(x, y, D, dt)

        rnew = hypot(xnew, ynew)

        # reflect off outer circle
        if rnew > R
            xnew, ynew = reflect_off_circle(xnew, ynew, R)
            rnew = hypot(xnew, ynew)
        end

        # reflect off inner circle (flagpole of radius a)
        if rnew < a
            # mirror across inner circle
            xnew, ynew = reflect_off_circle(xnew, ynew, a)
            rnew = hypot(xnew, ynew)
        end

        ang = atan(ynew, xnew)
        θ += angdiff(ang, prev_ang)
        prev_ang = ang

        x, y = xnew, ynew
    end

    return θ
end

# --- run many trajectories and plot ---

function main()
    Random.seed!(1234)     # reproducible

    N = 5000               # number of trajectories
    D = 1.0
    dt = 1e-3
    T  = 1200.0
    R  = 1.0
    a  = 0.2

    # case 1: pointlike
    θs_point = [simulate_winding_pointlike(D=D, dt=dt, T=T, R=R) for _ in 1:N]

    # case 2: finite-radius (annulus)
    θs_ann   = [simulate_winding_annulus(D=D, dt=dt, T=T, R=R, a=a) for _ in 1:N]

    # plot histograms side by side
    plt1 = histogram(θs_point, bins=60, normalize=true,
                     xlabel="θ", ylabel="P(θ)", title="Pointlike flagpole, finite R",
                     legend=false)
    plt2 = histogram(θs_ann, bins=60, normalize=true,
                     xlabel="θ", ylabel="P(θ)", title="Finite radius a, finite R",
                     legend=false)

    plot(plt1, plt2, layout=(1,2))
end

main()
