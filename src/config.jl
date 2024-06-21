module Config 

    using Parameters
    using NPZ
    using JSON
    export config_gen, config

    @with_kw struct config
        #* r-space indices for cluster (currently pair, triplet, quadruplet supported)
        ind_cluster::Dict{Any, Any}

        #* fractional coordinate of raw configuration
        ind_raw::Matrix{Float64}
        lattice_renorm::Matrix{Float64}

        #* cluster type list
        symbol_list::Vector{Any}

        #* basis vector / cluster cor func normalization (degeneracy and PBC) term 
        basis_norm_dict::Dict{Any, Any}
        cluster_norm_dict::Dict{Any, Any}
        # pbc_norm_cluster::Dict{Any, Any}
        
        #* kpoints in k-CE
        kpt_pairmesh::Matrix{Float64}
        kpt_tripmesh::Matrix{Float64}

        #* option: (always true in the provided model) ((not optional though
        normalize_basis::Bool = true
        normalize_cluster::Bool = true
        sym_cluster::Bool = true
        pbc_norm::Bool = true
    end

    function config_gen(atom_num)
        
        if !isdir("data/fcc/$(atom_num)")
            println("WARNING: cell's information not detected, default to 108")
            atom_num = 108
        end

        if atom_num > 256
            println("WARNING: cell size outside the interpolation range, result may be inaccurate")
        end

        ind_raw = npzread("data/fcc/$(atom_num)/ind_raw$(atom_num).npy")
        lattice_renorm = npzread("data/fcc/$(atom_num)/lattice_renorm$(atom_num).npy")
        basis_norm_dict = JSON.parsefile("data/cluster/basis_norm_dict_jl_quin.json")
        cluster_norm_dict = JSON.parsefile("data/cluster/cluster_norm_dict.json")
        # pbc_norm_dict = JSON.parsefile("data/fcc/$(atom_num)/norm_pbc_cluster_$(atom_num)_jl.json")
        #! unique clusters
        ind_cluster_dict = JSON.parsefile("data/fcc/$(atom_num)/cluster_ind_unic_$(atom_num)_jl.json")
        symbol_list = JSON.parsefile("./data/cluster/symbol_list_6nn.json")

        #* k-space CE part: load k-points
        kpt_pairmesh = npzread("data/kpoints/k_all_231209.npy")
        kpt_tripmesh = npzread("data/kpoints/non_zero_k_240403.npy")

        config_info = config(
            ind_cluster = ind_cluster_dict,
            ind_raw = ind_raw,
            lattice_renorm = lattice_renorm,
            symbol_list = symbol_list,
            basis_norm_dict = basis_norm_dict,
            cluster_norm_dict = cluster_norm_dict,
            # pbc_norm_cluster = pbc_norm_dict,
            kpt_pairmesh = kpt_pairmesh,
            kpt_tripmesh = kpt_tripmesh)

        return config_info
    end
end