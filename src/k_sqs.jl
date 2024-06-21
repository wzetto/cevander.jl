module k_sqs

    using LinearAlgebra
    using Random
    using IterTools
    using Statistics
    using StatsBase
    using Base.Threads
    using Random: randperm
    using Combinatorics
    using .GC

    include("utils.jl")
    include("config.jl")

    using .Utils
    using .Config
    export main_sqs_k

    function pointcpr_sqs(compo::Vector{Float64},
            spin_denote::Vector{Float64}=Vector{Float64}([2,1,-1,-2]))

        cpr_mat = hcat([phi_vec(spin) for spin in spin_denote]...)'
        return compo'*cpr_mat
    end

    function centre_pbc_make(cell_frac_, trans_vec=Array([0.,0.,0.]))
        cell_frac = copy(cell_frac_)
        cell_frac = cell_frac .- trans_vec'
        cell_frac = cell_frac .% 1.
        cell_frac_raw = deepcopy(cell_frac)

        pbc_vec = [
            [-1.,0,0],
            [0,-1,0],
            [-1,-1,0],
            [0,-1,-1],
            [-1,0,-1],
            [0,0,-1],
            [-1,-1,-1]]

        for vec_ in pbc_vec 
            cell_frac = vcat(cell_frac, cell_frac_raw .+ vec_')
        end 

        return cell_frac 
    end 

    function lattice_ft(k_list, origin_coord)

        """
        return N_k x L_sigma matrix
        """
        k_spin = sum(exp.(im.*((k_list*origin_coord'))), dims=2)/size(origin_coord, 1)
        return k_spin
    end

    function phi_buffer_sqs(compo, pairmesh, tripmesh, ind_real)

        σ_all = pointcpr_sqs(compo)
        σ_1, σ_2, σ_3, σ_4 = σ_all

        #* single k-point
        k_spin_1 = lattice_ft(pairmesh, ind_real)*σ_1
        k_spin_2 = lattice_ft(pairmesh, ind_real)*σ_2
        k_spin_3 = lattice_ft(pairmesh, ind_real)*σ_3
        k_spin_4 = lattice_ft(pairmesh, ind_real)*σ_4

        k_pair = ([
            k_spin_1.*conj.(k_spin_1),
            k_spin_2.*conj.(k_spin_2),
            k_spin_3.*conj.(k_spin_3),
            k_spin_4.*conj.(k_spin_4),

            (k_spin_1.*conj.(k_spin_2) .+ k_spin_2.*conj.(k_spin_1))/2,
            (k_spin_1.*conj.(k_spin_3) .+ k_spin_3.*conj.(k_spin_1))/2,
            (k_spin_1.*conj.(k_spin_4) .+ k_spin_4.*conj.(k_spin_1))/2,
            (k_spin_2.*conj.(k_spin_3) .+ k_spin_3.*conj.(k_spin_2))/2,
            (k_spin_2.*conj.(k_spin_4) .+ k_spin_4.*conj.(k_spin_2))/2,
            (k_spin_3.*conj.(k_spin_4) .+ k_spin_4.*conj.(k_spin_3))/2,
        ])

        k_pair = vcat(k_pair...)'

        #* 3-body part
        dim_tripmat = Int64(size(tripmesh, 1)*(size(tripmesh, 1)-1)/2)
        trip_1, trip_2, trip_3 = zeros((dim_tripmat, 3)), zeros((dim_tripmat, 3)), zeros((dim_tripmat, 3))
        corfunc_trip = zeros(0)
        for (count, (i_k1, i_k2)) in enumerate(combinations(1:size(tripmesh, 1), 2))
            k1, k2 = tripmesh[i_k1,:], tripmesh[i_k2,:]
            trip_1[count, :] .= k1
            trip_2[count, :] .= k2
            trip_3[count, :] .= k1 .+ k2
        end 

        for (i, j, k) in product(1:4, 1:4, 1:4) #* for each comb. of basis function
            k_spin_1 = lattice_ft(trip_1, ind_real)*σ_all[i]
            k_spin_2 = lattice_ft(trip_2, ind_real)*σ_all[j]
            k_spin_3 = conj.(lattice_ft(trip_3, ind_real))*σ_all[k]
    
            k_corfunc = k_spin_1.*k_spin_2.*k_spin_3
            corfunc_trip = vcat(corfunc_trip, k_corfunc)
        end 

        return real.(k_pair), corfunc_trip', (trip_1, trip_2, trip_3)
    end

    function main_sqs_k(
                        atom_num::Int64,
                        compo_list::Matrix{Float64},)

        config_info = config_gen(atom_num)

        ind_raw_ = config_info.ind_raw
        pairmesh, tripmesh = config_info.kpt_pairmesh, config_info.kpt_tripmesh
        atom_num = size(ind_raw_, 1)
        compo_num = size(compo_list, 1)

        ind_raw_pbc = centre_pbc_make(ind_raw_)
        lattice_renorm = config_info.lattice_renorm
        ind_raw_pbc = ind_raw_pbc .* lattice_renorm

        max_coord, min_coord = lattice_renorm/2 .-1e-3, -lattice_renorm/2 .-1e-3
        ind_prim = [i for i in 1:size(ind_raw_pbc, 1) if all(min_coord .< ind_raw_pbc[i,:] .< max_coord)]
        ind_raw = ind_raw_pbc[ind_prim, :]

        cpr_mat_k = [Vector{Float64}() for _ in 1:compo_num]
        for compo_i in 1:size(compo_list, 1)

            compo_ = compo_list[compo_i,:]

            phi_pair, phi_trip, tripmesh_sqs = phi_buffer_sqs(compo_, pairmesh, tripmesh, ind_raw)
            trip_real, trip_imag = real.(phi_trip), -imag.(phi_trip)

            weight_k_FIN = hcat(phi_pair, trip_real, trip_imag)
            cpr_mat_k[compo_i] = vec(weight_k_FIN)
        end

        return cpr_mat_k
    end

end
