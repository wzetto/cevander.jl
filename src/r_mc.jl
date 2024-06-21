module r_mc

    using Random
    using LinearAlgebra
    using NPZ
    using Combinatorics
    using Statistics
    using JLD2
    using Distributed
    using Base.Threads
    using Base
    using Parameters

    include("utils.jl")
    include("config.jl")

    using .Utils
    using .Config

    export main_mc_r

    function cluster_symoperator(config_info::config, 
                                cluster_type::String,
                                spin_list::Vector{Float64},
                                spin_map_dict=Dict{Int64, String}(1 => "p", -1 => "n"))
        """
        return degeneracy term of cluster correlation function
        """
        cluster_norm_dict = config_info.cluster_norm_dict
        spin_denote = join([spin_map_dict[spin_list[i]/abs(spin_list[i])]*"$(Int(abs(spin_list[i])))" for i in 1:length(spin_list)], "")
        sym_cluster = cluster_norm_dict[cluster_type][spin_denote]
        return sym_cluster
    end


    function basis_symoperator(config_info::config,
                            cluster_type::String)
        """
        return degeneracy term of basis correlation function
        """
        basis_norm_dict_ = config_info.basis_norm_dict
        norm_basis_ind = basis_norm_dict_[cluster_type]
        return norm_basis_ind
    end

    function cpr(spin_list::Vector{Float64}, cluster_type::String, config_info::config)
        """
        return cluster correlation function for one cluster
        """
        outer_prod = kron([phi_vec(spin) for spin in spin_list]...)
        cluster_degeneracy = cluster_symoperator(config_info, cluster_type, spin_list)

        return outer_prod / cluster_degeneracy
    end

    function point_cpr(embed_list::Vector{Float64})
        return phi_vec(embed_list)
    end

    function cpr_unique(spin_input_raw::Vector{Vector{Float64}},
                        cluster_type::String,
                        config_info::config,
                        norm_list::Vector{Float64})::Vector{Float64}

        unique_spin = unique(spin_input_raw, dims=1)
        len_unic = size(unique_spin, 1)
        cpr_vec = 0
        for i in 1:len_unic
            spin_ = unique_spin[i, :]

            match_spin = all(spin_input_raw .== spin_, dims=2)
            ind_spin = findall(x -> x == true, match_spin)

            spin_corfunc = cpr(spin_[1], cluster_type, config_info)
            cpr_vec = cpr_vec .+ spin_corfunc * sum(norm_list[ind_spin])
        end

        #* symmetry normalization of basis vector
        basis_mergeind = basis_symoperator(config_info, cluster_type)
        len_basis = length(basis_mergeind)
        cpr_vec = Vector{Float64}([mean(cpr_vec[basis_mergeind[i]]) for i ∈ 1:len_basis])

        return cpr_vec
    end

    function cluster_extract(embed_list::Vector{Float64},
        cluster_type_list::Vector{Any},
        config_info::config)
        """
        return cluster correlation function for all clusters
        """
        cpr_vec = [Vector{Float64}() for _ in 1:length(cluster_type_list)]
        len_symbol = length(cluster_type_list)
        
        for count in 1:len_symbol
            cluster_type = cluster_type_list[count]
            cluster_ind = config_info.ind_cluster[cluster_type]
            
            #* pbc norm term: <\Phi(\alpha)/(N_PBC > n_atom + 1)>/g(\alpha)
            norm_list = ones(length(cluster_ind))
            spin_list_buffer_all = [embed_list[ind] for ind in cluster_ind]
            
            cpr_return  = cpr_unique(spin_list_buffer_all, cluster_type, config_info, norm_list)

            #* [\Phi(1/(N+1))^T]\sum 1/(N+1)
            cpr_fin = cpr_return / sum(norm_list)
            cpr_vec[count] = cpr_fin

        end
        #* flatten cpr_vec
        cpr_vec = reduce(vcat, cpr_vec)
        append!(cpr_vec, point_cpr(embed_list))

        return cpr_vec
    end

    function main_mc_r(iter::UnitRange{Int64}, 
                        atom_num::Int64,
                        embed_list_raw::Matrix{Float64})

        config_info = config_gen(atom_num)

        cpr_mat = [Vector{Float64}() for _ in 1:length(iter)]
        Threads.@threads for embed_i in iter
            embed_list = repeat(embed_list_raw[embed_i, :], 27)
            key_list_ = config_info.symbol_list
            cpr_vec = cluster_extract(embed_list, key_list_, config_info)

            cpr_mat[embed_i] = cpr_vec

            if embed_i % 100 == 0
                GC.gc()
            end
        end

        return cpr_mat
    end

end