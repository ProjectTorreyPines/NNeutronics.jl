module NNeutronics

using CSV
using DataFrames
using Interpolations
using Printf

mutable struct Blanket{
    T<:Interpolations.AbstractExtrapolation,
    L<:Interpolations.AbstractExtrapolation,
    H<:Interpolations.AbstractExtrapolation,
    F<:Interpolations.AbstractExtrapolation
}
    D1::Vector{Float64}
    D2::Vector{Float64}
    D3::Vector{Float64}
    enrichment::Vector{Float64}
    energy_grid::Vector{Float64}
    radial_grid::Vector{Float64}
    TBR::Array{Float64,4}
    leakeage_energy::Array{Float64,5}
    heating::Array{Float64,5}
    flux::Array{Float64,5}
    i_log10_TBR::T
    i_log10_leakeage_energy::L
    i_log10_heating::H
    i_log10_flux::F
end

Base.broadcastable(blanket::Blanket) = (blanket,)

"""
    TBR(blanket::Blanket, d1::Union{<:Real,Vector{<:Real}}, d2::Union{<:Real,Vector{<:Real}}, d3::Union{<:Real,Vector{<:Real}}, Li6::Union{<:Real,Vector{<:Real}})

Interpolation in d1, d2, d3, Li6 for TBR
"""
function TBR(blanket::Blanket, d1::Union{<:Real,Vector{<:Real}}, d2::Union{<:Real,Vector{<:Real}}, d3::Union{<:Real,Vector{<:Real}}, Li6::Union{<:Real,Vector{<:Real}})
    return 10.0 .^ (blanket.i_log10_TBR.(d1, d2, d3, Li6)) .- 1E-16
end

"""
    TBR(blanket::Blanket, d1::Union{<:Real,Vector{<:Real}}, d2::Union{<:Real,Vector{<:Real}}, d3::Union{<:Real,Vector{<:Real}}, Li6::Union{<:Real,Vector{<:Real}})

Interpolation in d1, d2, d3, Li6 for leakeage energy spectrum
"""
function leakeage_energy(
    blanket::Blanket,
    d1::Union{<:Real,Vector{<:Real}},
    d2::Union{<:Real,Vector{<:Real}},
    d3::Union{<:Real,Vector{<:Real}},
    Li6::Union{<:Real,Vector{<:Real}},
    energy::Vector{Float64}=blanket.energy_grid
)
    return 10.0 .^ (blanket.i_log10_leakeage_energy.(d1, d2, d3, Li6, energy)) .- 1E-16
end

"""
    heating(blanket::Blanket, d1::Union{<:Real,Vector{<:Real}}, d2::Union{<:Real,Vector{<:Real}}, d3::Union{<:Real,Vector{<:Real}}, Li6::Union{<:Real,Vector{<:Real}})

Interpolation in d1, d2, d3, Li6 for volumetric heating
"""
function heating(
    blanket::Blanket,
    d1::Union{<:Real,Vector{<:Real}},
    d2::Union{<:Real,Vector{<:Real}},
    d3::Union{<:Real,Vector{<:Real}},
    Li6::Union{<:Real,Vector{<:Real}},
    radius::Vector{Float64}=blanket.radial_grid
)
    return 10.0 .^ (blanket.i_log10_heating.(d1, d2, d3, Li6, radius)) .- 1E-16
end

"""
    flux(blanket::Blanket, d1::Union{<:Real,Vector{<:Real}}, d2::Union{<:Real,Vector{<:Real}}, d3::Union{<:Real,Vector{<:Real}}, Li6::Union{<:Real,Vector{<:Real}})

Interpolation in d1, d2, d3, Li6 for flux
"""
function flux(
    blanket::Blanket,
    d1::Union{<:Real,Vector{<:Real}},
    d2::Union{<:Real,Vector{<:Real}},
    d3::Union{<:Real,Vector{<:Real}},
    Li6::Union{<:Real,Vector{<:Real}},
    radius::Vector{Float64}=blanket.radial_grid
)
    return 10.0 .^ (blanket.i_log10_flux.(d1, d2, d3, Li6, radius)) .- 1E-16
end

function energy_edges()
    return [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
end

function energy_grid()
    energy = energy_edges()
    return (energy[1:end-1] + energy[2:end]) / 2.0
end

function radial_edges()
    radial = [1000.0, 1003.2175, 1006.435, 1009.6525, 1012.87, 1016.0875, 1019.305, 1022.5225, 1025.74, 1028.9575, 1032.175, 1035.3925, 1038.61, 1041.8275, 1045.045, 1048.2625, 1051.48, 1054.6975, 1057.915, 1061.1325, 1064.35, 1067.5675, 1070.785, 1074.0025, 1077.22, 1080.4375, 1083.655, 1086.8725, 1090.09, 1093.3075, 1096.525, 1099.7425, 1102.96, 1106.1775, 1109.395, 1112.6125, 1115.83, 1119.0475, 1122.265, 1125.4825, 1128.7]
    return (radial .- radial[1]) ./ (radial[end] .- radial[1])
end

function radial_grid()
    radial = radial_edges()
    return (radial[1:end-1] + radial[2:end]) / 2.0
end

function col_names()
    return vcat(["d1", "d2", "d3", "Li6", "TBR"],
        map(x -> @sprintf("escape_%3.3f", x), energy_grid()),
        map(x -> @sprintf("heating_%3.3f", x), radial_grid()),
        map(x -> @sprintf("flux_%3.3f", x), radial_grid()))
end

"""
    Blanket()

Returns a `Blanket` structure
"""
function Blanket()
    df = DataFrame(CSV.File(dirname(@__FILE__) * "/../data/NNeutronics_DB.csv"))
    sort!(df, [:Li6, :d3, :d2, :d1])

    D1 = sort(unique(df.d1))
    D2 = sort(unique(df.d2))
    D3 = sort(unique(df.d3))
    enrichment = sort(unique(df.Li6))

    TBR = Array{Float64}(undef, size(D1)..., size(D2)..., size(D3)..., size(enrichment)...)
    for (k, row) in enumerate(eachrow(df))
        TBR[k] = row.TBR
    end

    tmp = Array{Float64}(undef, size(D1)..., size(D2)..., size(D3)..., size(enrichment)...)

    leakeage_energy = Array{Float64}(undef, size(D1)..., size(D2)..., size(D3)..., size(enrichment)..., size(energy_grid())...)
    for (ke, col) in enumerate(names(df)[6:6-1+length(energy_grid())])
        for (k, row) in enumerate(eachrow(df))
            tmp[k] = getproperty(row, col)
        end
        leakeage_energy[:, :, :, :, ke] = tmp
    end

    heating = Array{Float64}(undef, size(D1)..., size(D2)..., size(D3)..., size(enrichment)..., size(radial_grid())...)
    for (kr, col) in enumerate(names(df)[6+length(energy_grid()):6-1+length(energy_grid())+length(radial_grid())])
        for (k, row) in enumerate(eachrow(df))
            tmp[k] = getproperty(row, col)
        end
        heating[:, :, :, :, kr] = tmp
    end

    flux = Array{Float64}(undef, size(D1)..., size(D2)..., size(D3)..., size(enrichment)..., size(radial_grid())...)
    for (kr, col) in enumerate(names(df)[6+length(energy_grid())+length(radial_grid()):6-1+length(energy_grid())+2*length(radial_grid())])
        for (k, row) in enumerate(eachrow(df))
            tmp[k] = getproperty(row, col)
        end
        flux[:, :, :, :, kr] = tmp
    end

    i_log10_TBR = extrapolate(interpolate((D1 / 100, D2 / 100, D3 / 100, enrichment), log10.(TBR .+ 1E-16), Gridded(Linear())), Flat())
    i_log10_leakeage_energy = extrapolate(interpolate((D1 / 100, D2 / 100, D3 / 100, enrichment, energy_grid()), log10.(leakeage_energy .+ 1E-16), Gridded(Linear())), Flat())
    i_log10_heating = extrapolate(interpolate((D1 / 100, D2 / 100, D3 / 100, enrichment, radial_grid()), log10.(heating .+ 1E-16), Gridded(Linear())), Flat())
    i_log10_flux = extrapolate(interpolate((D1 / 100, D2 / 100, D3 / 100, enrichment, radial_grid()), log10.(flux .+ 1E-16), Gridded(Linear())), Flat())

    return Blanket(
        D1 / 100,
        D2 / 100,
        D3 / 100,
        enrichment,
        energy_grid(),
        radial_grid(),
        TBR,
        leakeage_energy,
        heating,
        flux,
        i_log10_TBR,
        i_log10_leakeage_energy,
        i_log10_heating,
        i_log10_flux
    )
end

export Blanket, TBR, leakeage_energy, heating, flux

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__, all=false, imported=false) if name != Symbol(@__MODULE__)]

end # module
