module Utils

    using Random
    using Statistics
    using LinearAlgebra

    export phi_vec, ele_list_gen, eval_mix, ϕ1, ϕ2, ϕ3, ϕ4, vec2str

    #* basis function for 5-dimension vector space (J. Phys. Soc. Jpn. 88, 044803 (2019))
    #* this aims to construct a 4-dimension subspace to avoid the zero-expectation value problem

    function ϕ1(σ)
        return sqrt(2)/2 .* σ
    end

    function ϕ2(σ)
        return -sqrt(10/7) .+ sqrt(5/14) .* σ.^2
    end

    function ϕ3(σ)
        return -17/(6*sqrt(2)).*σ .+ 5/(6*sqrt(2)).*σ.^3
    end

    function ϕ4(σ)
        return 3*sqrt(14)/7 .- 155*sqrt(14)/168*σ.^2 .+ 5*sqrt(14)/24*σ.^4
    end

    function phi_vec(σ::Float64)
        return Vector{Float64}([ϕ1(σ), ϕ2(σ), ϕ3(σ), ϕ4(σ)])
    end

    function phi_vec(σ::Vector{Float64})
        return Vector{Float64}([mean(ϕ1(σ)), mean(ϕ2(σ)), mean(ϕ3(σ)), mean(ϕ4(σ))])
    end

    function ele_list_gen(compo_list::Vector{Float64},
                        num_cell=108,
                        cell_sequence_num::Int64=1,
                        mode::String="randchoice")
        @assert abs(sum(compo_list)-1) < 0.001 "Make sure the atomic contents sum up to 1"

        """
        Recommend to use mode "int" if the expected number of composed elements are all in integer
        """
        
        if !isdir("data/fcc/$(num_cell)")
            println("WARNING: structural information not detected, default to 108-atom cell")
            num_cell = 108
        end

        if num_cell > 256
            println("WARNING: cell size outside the interpolation range, result may be inaccurate")
        end

        spin_seq = [2, 1, -1, -2]
        seq_buffer = zeros(Float64, cell_sequence_num, num_cell)
        for i in 1:cell_sequence_num
            while true
                if cmp("randchoice", mode) == 0
                    num_seq = [rand(range(convert(Int, round(c*num_cell)), 
                               convert(Int, round(c*num_cell))+1, step=1)) for c in compo_list[1:3]]
                elseif cmp("int", mode) == 0
                    num_seq = [convert(Int, round(c*num_cell)) for c in compo_list[1:3]]
                end

                len_ni = num_cell - sum(num_seq)
                if abs(len_ni-num_cell*compo_list[4]) <= 1
                    append!(num_seq, len_ni)
                    ele_list_raw = cat([zeros(Float64, num_seq[i]) .+ spin_seq[i] for i in 1:4]..., dims=1)
                    seq_buffer[i,:] = shuffle(ele_list_raw)
                    break
                end
            end
        end
        return seq_buffer
    end

    function eval_mix(θ, X)
        N = size(X, 1)
        𝐗 = hcat(X, ones(N))
        y_pred = 𝐗 * θ
        return y_pred
    end

    function vec2str(compo_vec)
        compo_denote = join(convert(Vector{Int64}, round.(compo_vec*100),), "")
        if compo_vec[1]+compo_vec[2] > 0.5 || maximum(compo_vec.-0.5) > 0
            println("WARNING: composition outside the interpolation range, result may be inaccurate")
        end
        return compo_denote
    end
end