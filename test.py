# import jax.numpy as jnp
# from jax import vmap

# vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
# mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
# mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

# a = jnp.array([1,1,1])
# b = jnp.array([2,1,0])
# A = a[None]
# print(A)
# print(mv(A,b))


# Current initialization
S, S_props = zip(*[hmm.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)])
S_l, S_l_props = zip(*[hmm_n.initialize(jr.PRNGKey(key)) for key in range(STUDENTS_NUM)])

# Training function
fit = lambda hmm_class, params, props, emissions : [hmm_class.fit_em(param, prop, emissions) for param, prop in zip(params, props)]

# Current training calls
S0, _   = fit(hmm, S, S_props, T0_emissions_train)
S1, _   = fit(hmm, S, S_props, T1_emissions_train)
S00, _  = fit(hmm, S0, S_props, T0_emissions_train)
S01, _  = fit(hmm, S0, S_props, T1_emissions_train)
S11, _  = fit(hmm, S1, S_props, T1_emissions_train)
T01, _  = fit(hmm, T0, T0_props, T1_emissions_train)
S_l0, _ = fit(hmm_n, S_l, S_l_props, T0_emissions_train)

# Evaluation function
evaluate_func = lambda hmm_class : vmap(hmm_class.marginal_log_prob, [None, 0], 0)
ev = lambda hmm, features, test: (evaluate_func(hmm)(features, test)).mean()

# Results compilation
params = [
    ["T0" , T0 , hmm],
    ["T1" , T1 , hmm],
    ["T2" , T2 , hmm], 
    ["S"  , S  , hmm],
    ["S0" , S0 , hmm],
    ["S1" , S1 , hmm],
    ["S00", S00, hmm],
    ["S01", S01, hmm],
    ["S11", S11, hmm],
    ["T01" , T01, hmm],
    ["S_l", S_l, hmm_n],
    ["S_l0",S_l0, hmm_n]
]

# Results processing
for key, models, hmm_type in params:
    for model in models:
        results[key].append([float((ev(hmm_type, model, test)-base(train, test))/(ev(hmm, T, test)-base(train, test))) for T, train, test in teachers])