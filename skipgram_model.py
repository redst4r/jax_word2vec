import jax.numpy as jnp
import jax

@jax.jit
def _helper_word2vec_loss_negative_sampling(params, target, neg_samples, context):
    """Compute the loss of the word2Vec model."""
    
    context_rep = params["projection"][context]     #(batchsize, C,  D)
    target_rep = params["projection"][target]   #(batchsize, D)

    # prediction for each context
    logits_pos = jnp.einsum('bcd,bd->bc', context_rep, target_rep)  # batch, C
    pos_loss_per_Sample = -jax.nn.log_sigmoid(logits_pos).sum(axis=1)  # summing the loss over all contexts per sample

    negative_target_embeddings = params["projection"][neg_samples]  # batch x n_neg x dim
    negative_logits = jnp.einsum("bd,bnd->bn", target_rep, negative_target_embeddings)   # batch x n_neg
    neg_loss_per_Sample = -jax.nn.log_sigmoid(-negative_logits).sum(axis=1)  # mind the -1 in the sigmoid! also we sum over all neg samples

    pos_loss = pos_loss_per_Sample.mean()
    neg_loss = neg_loss_per_Sample.mean()
    
    return pos_loss, neg_loss
    
@jax.jit
def word2vec_loss_negative_sampling(params, target, neg_samples, context):
    pos_loss, neg_loss = _helper_word2vec_loss_negative_sampling(params, target=target, neg_samples=neg_samples, context=context)
    return pos_loss + neg_loss


"""
the non-batched versions
"""

@jax.jit
def _helper_word2vec_loss_negative_sampling_single_sample(params, target, neg_samples, context):
    """
    dimensionas:
    C: number of context
    D: latent dim
    N: number of neg examples
    
    target: shape: int
    context: shape: int[C]
    neg_samples: shape: int[Neg]
    """
    context_rep = params["projection"][context]     #(C,  D)
    target_rep = params["projection"][target]    # (D,)

    logits_pos = jnp.einsum('cd,d->c', context_rep, target_rep)  # C
    pos_loss_per_Sample = -jax.nn.log_sigmoid(logits_pos).sum()  # summing the loss over all contexts per sample

    
    negative_target_embeddings = params["projection"][neg_samples]  # n_neg x dim
    negative_logits = jnp.einsum("d,nd->n", target_rep, negative_target_embeddings)   # batch x n_neg
    neg_loss_per_Sample = -jax.nn.log_sigmoid(-negative_logits).sum()  # mind the -1 in the sigmoid! also we sum over all neg samples

    return neg_loss_per_Sample + pos_loss_per_Sample

@jax.jit
def word2vec_loss_negative_sampling_single_sample(params, target_batch, neg_samples_batch, context_batch):
    """
    feed in batches, but uses vmap to calculate acrsoss batches
    """
    f_batch = jax.vmap(_helper_word2vec_loss_negative_sampling_single_sample, in_axes=[None, 0,0,0])
    return f_batch(params, target_batch, neg_samples_batch, context_batch).mean()







