from macros import *
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
import pandas as pd
import seaborn as sns
from dynamax.hidden_markov_model import GaussianHMM
import jax.numpy as jnp



def sample_hmm(hmm_model,length=40,trials=400,seed_base=0,flatten=False, return_states=True):

    X = [hmm_model.sample(length,random_state=check_random_state(i+seed_base)) for i in range(trials)]
    obs,state = zip(*X)

    #lengths = [x.shape[0] for x in X]
    obs = np.stack(obs)
    state = np.stack(state)[...,None]
    if flatten:
        obs = obs.flatten()[..., None]
        state = state.flatten()[...,None]
    if return_states:
        return obs,state
    return obs



def score_hmm(hmm_model,observations):
    X = jnp.concatenate([observations[i] for i in range(observations.shape[0])],axis=0)
    lengths = [observations.shape[1]] * observations.shape[0]
    return hmm_model.score(X,lengths)



# def predict_proba(hmm_model,observations,back_to_3D=True):
#     X = jnp.concatenate([observations[i] for i in range(observations.shape[0])],axis=0)
#     lengths = [observations.shape[1]] * observations.shape[0]
#     proba = hmm_model.predict_proba(X, lengths)
#     if back_to_3D:
#         proba = proba.reshape(*observations.shape[:2],proba.shape[-1])
#     return proba


def predict_proba(hmm, params, observations, back_to_3D=True):
    proba_list = []
    for obs in observations:
        # Run the smoother to get marginal probabilities over hidden states
        posteriors = hmm.smoother(params, obs)
        # Extract numerical probabilities from the posterior object
        proba_list.append(posteriors.smoothed_probs)  # or posteriors.marginal_probs
    
    # Ensure all elements are arrays before concatenating
    proba_list = [jnp.array(p) for p in proba_list]
    proba = jnp.concatenate(proba_list, axis=0)

    if back_to_3D:
        proba = proba.reshape(*observations.shape[:2], -1)
    return proba




def add_ring(model,ring_length=2,eps=0):
    new_model = deepcopy(model)
    new_model.n_components = model.n_components * ring_length
    # ring_transmat = jnp.roll(jnp.eye(ring_length),1,axis=1)
    ring_transmat = jnp.eye(ring_length,k=1)
    new_model.transmat_ = jnp.kron(model.transmat_,ring_transmat) + eps
    new_model.transmat_ /= new_model.transmat_.sum(axis=1,keepdims=True)
    # new_model.startprob_ = jnp.stack([model.startprob_]*ring_length).reshape(new_model.n_components) / ring_length
    new_model.startprob_ = jnp.kron(
        model.startprob_,
        (jnp.arange(ring_length)==0).astype(float)
    )  #jnp.stack([model.startprob_] * ring_length).reshape(new_model.n_components) / ring_length
    # new_model.n_states = model.n_states * ring_length
    print(new_model.means_.shape,new_model.covars_.shape)
    new_model.means_ = jnp.concatenate([model.means_] * ring_length,axis=0)
    covars = model.covars_
    if model.covariance_type == "diag":
        covars = jnp.einsum('ijj->ij',covars)
    new_model.covars_ = jnp.concatenate([covars] * ring_length, axis=0)
    return new_model


def add_ring(model,ring_length=2,eps=1e-5):
    new_model = deepcopy(model)
    new_model.n_components = model.n_components * ring_length
    # ring_transmat = jnp.roll(jnp.eye(ring_length),1,axis=1)
    ring_transmat = jnp.eye(ring_length,k=1)
    new_model.transmat_ = jnp.kron(ring_transmat, model.transmat_) + eps
    new_model.transmat_ /= new_model.transmat_.sum(axis=1,keepdims=True)
    new_model.startprob_ = jnp.stack([model.startprob_]*ring_length).reshape(new_model.n_components) / ring_length
    # new_model.n_states = model.n_states * ring_length
    print(new_model.means_.shape,new_model.covars_.shape)
    new_model.means_ = jnp.concatenate([model.means_] * ring_length,axis=0)
    covars = model.covars_
    if model.covariance_type == "diag":
        covars = jnp.einsum('ijj->ij',covars)
    new_model.covars_ = jnp.concatenate([covars] * ring_length, axis=0)
    return new_model



# def predict_proba(hmm_model, observations, back_to_3D=False):
#     # Compute the posterior state probabilities
#     log_likelihood, posterior_probs = hmm_model.forward_backward(observations)
#     return posterior_probs



batch_choice = lambda p:  np.stack([np.random.choice(len(p[i]),p=p[i]) for i in range(len(p))])

# def decoding(input_hmm, input_model, target_hmm, target_model, observations, name_to_save_plot="test.png"):
#     train_observations,test_observations = train_test_split(observations)

#     input_proba = predict_proba(input_hmm, input_model, train_observations, back_to_3D=False)
#     target_proba = predict_proba(target_hmm, target_model, train_observations, back_to_3D=False)

#     target_proba = np.array(target_proba, dtype=np.float32)
#     target_proba /= target_proba.sum(axis=1, keepdims=True)


#     target_proba_sampled = batch_choice(target_proba)

#     # TO TEST SEVERAL MODELS
#     # model = LogisticRegression(multi_class='multinomial') 
#     # models = [LogisticRegression(),MLPClassifier(hidden_layer_sizes=(100,)),MLPClassifier(hidden_layer_sizes=(200,))]
#     # model_names = ['LogisticRegression','MLPClassifier(hidden_layer_sizes=(100,))','MLPClassifier(hidden_layer_sizes=(200,))']
#     # for model in models:
#     #     model.fit(
#     #         input_proba,
#     #         target_proba_sampled
#     #     )

#     model = MLPClassifier(hidden_layer_sizes=(200,), )
#     model.fit(input_proba,target_proba_sampled)

#     ### testing
#     input_proba = predict_proba(input_hmm, input_model, test_observations, back_to_3D=False)
#     target_proba = predict_proba(target_hmm, target_model, test_observations, back_to_3D=False)
#     # print(input_proba.shape,target_proba.shape)

#     # print(pred_proba[:10,:])
#     # print(pred_target[:3])

#     # import matplotlib.pyplot as plt
#     # import os
#     # fig,axs = plt.subplots(3,1,sharex=True)
#     # axs[0].imshow(pred_proba[:10].T, aspect='auto',vmax=1)
#     # axs[1].imshow(target_proba[:10].T, aspect='auto',vmax=1)
#     # axs[2].imshow(input_proba[:10].T, aspect='auto', vmax=1)
#     # plt.savefig(os.path.join('plots',name_to_save_plot))
#     # plt.close(fig)

#     # TO TEST SEVERAL MODELS
#     # scores = {}
#     # for model_name,model in zip(model_names,models):
#     #     pred_proba = model.predict_proba(input_proba)
#         # metric = rel_entr
#         # score = np.stack([metric(
#         #     target_proba[i],pred_proba[i]
#         # ) for i in range(pred_proba.shape[0])]).mean()
#         # scores[model_name] = score
#     pred_proba = model.predict_proba(input_proba)
#     # print(input_proba.shape, target_proba.shape, pred_proba.shape)
    
#     metric = rel_entr
#     score = np.stack([metric(
#         target_proba[i],pred_proba[i]
#     ) for i in range(pred_proba.shape[0])]).mean()

#     return score


def decoding(input_hmm, input_model, target_hmm, target_model, observations, name_to_save_plot="test.png"):
    # Splitting observations
    train_observations, test_observations = train_test_split(observations)

    # Get probabilities for training and testing
    input_proba = predict_proba(input_hmm, input_model, train_observations, back_to_3D=False)
    target_proba = predict_proba(target_hmm, target_model, train_observations, back_to_3D=False)

    # Normalize target probabilities
    target_proba = np.array(target_proba, dtype=np.float32)
    target_proba /= target_proba.sum(axis=1, keepdims=True)
    
    # Generate sampled labels
    target_proba_sampled = batch_choice(target_proba)

    # Model training
    model = MLPClassifier(hidden_layer_sizes=(200,))
    model.fit(input_proba, target_proba_sampled)

    ### Testing
    input_proba = predict_proba(input_hmm, input_model, test_observations, back_to_3D=False)
    target_proba = predict_proba(target_hmm, target_model, test_observations, back_to_3D=False)

    pred_proba = model.predict_proba(input_proba)

    # Handling unseen classes
    all_classes = np.arange(target_proba.shape[-1])  # Assume classes are [0, 1, ..., num_classes - 1]
    full_pred_probs = np.zeros((pred_proba.shape[0], len(all_classes)))

    # Map predicted probabilities to the correct indices
    for i, cls in enumerate(model.classes_):
        full_index = np.where(all_classes == cls)[0][0]
        full_pred_probs[:, full_index] = pred_proba[:, i]

    full_pred_probs += 1e-7
    full_pred_probs /= full_pred_probs.sum(axis=1, keepdims=True)

    # Compute the relative entropy score
    metric = rel_entr
    score = np.stack([
        metric(target_proba[i], full_pred_probs[i])
        for i in range(full_pred_probs.shape[0])
    ]).mean()

    return score





if __name__ == '__main__':
    model = GaussianHMM(n_components=2, random_state=0)

    # randomly initialising with 10D observable
    seq_length = 10
    D = 10
    trials = 5000 #00
    model._init(jnp.array(np.random.normal(size=(seq_length,D))),[seq_length])
    # model.transmat_ = jnp.roll(jnp.eye(2),-1,axis=0)
    observations,states = sample_hmm(model,length=seq_length,trials=trials,flatten=False,return_states=True)
    print('observation shape',observations.shape)
    print('states shape', states.shape)

    new_model = add_ring(model, ring_length=4)
    param_names = ['transmat_','startprob_','means_','covars_']
    print('model param shapes', {pname:getattr(model,pname).shape for pname in param_names})
    print('new_model param shapes',{pname:getattr(new_model, pname).shape for pname in param_names})



    print("model score", score_hmm(model, observations))
    print("new_model score", score_hmm(new_model, observations))


    # print(
    #     "\ntransmat\n",new_model.transmat_,
    #     "\nstartprob\n", new_model.startprob_,
    #     "\nmeans\n", new_model.means_,
    #     "\ncovars\n", new_model.covars_
    #
    # )
    all_scores = pd.DataFrame({
        'model to model'     : decoding(model, model, observations, "model_to_model.png"),
        "model to new_model" : decoding(model,new_model,observations, "model_to_newmodel.png"),
        "new_model to model" : decoding(new_model, model, observations, "newmodel_to_model.png")
    })

    print(all_scores)

    # all_scores.T.to_csv("plots/all_test_scores.csv")



