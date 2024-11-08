module cevander

    export phi_vec, ele_list_gen, eval_mix, ϕ1, ϕ2, ϕ3, ϕ4, vec2str
    export config_gen, config
    export main_mc_k, main_sqs_k, main_mc_r, main_sqs_r

    include("utils.jl")
    include("k_mc.jl")
    include("r_mc.jl")
    include("r_sqs.jl")
    include("k_sqs.jl")
    include("config.jl")

    using .Utils
    using .k_sqs 
    using .r_sqs
    using .k_mc
    using .r_mc

end