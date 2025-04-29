import jax.numpy as jnp
import jax
import optax

"""
skipgram softmax
"""
@jax.jit
def skipgram_softmax_forward(params,  target):
    """Forward pass of the skipgram model.
    do the inner product of target and each word in the vocab

    context is a (batch_size, 2*window_size) array of word IDs.

    V is the vocabulary size, D is the embedding dimension.
    params["projection"] is a (V, D) matrix of word embeddings.
    """
    # Indexing into (V, D) matrix with a batch of IDs. The output shape
    # is (batch_size, D).
    projection = params["projection"][target]  # batch, dim

    # all targets
    projection_target = params["projection"]   # batch, vocab

    hidden = jnp.einsum('bd,vd->bv', projection, projection_target)  # batch, vocab
    return hidden

@jax.jit
def skipgram_softmax_loss(params, target, context):
    """Compute the loss of the word2Vec model."""
    V, D = params['projection'].shape

    logits = skipgram_softmax_forward(params, target)  # batch, vocab

    # those are the predictions given the center word
    # now check how those work against all different words in the context
    # softmax along the second dimension
    context_onehot = jax.nn.one_hot(context, V)  # (batch_size, C, V)

    
    # naive softmax XE
    if False:
        loss_per_sample = jax.lax.fori_loop(
            0, context_onehot.shape[1], init_val=jnp.zeros(context_onehot.shape[0]),
            body_fun= lambda i, acc: acc+optax.losses.softmax_cross_entropy(logits, context_onehot[:,i,:], axis=1)
            
        )
    else:
        # however, we're reusing hte same logits all the time, only the targets change!
        """literally copeid from optax.losses.softmax_cross_entropy
        chex.assert_type([logits], float)
        log_probs = jax.nn.log_softmax(logits, axis, where)
        return -(labels * log_probs).sum(axis, where=where)
        """
        log_probs = jax.nn.log_softmax(logits, axis=1)
        loss_per_sample = jax.lax.fori_loop(
            0, context_onehot.shape[1], init_val=jnp.zeros(context_onehot.shape[0]),
            body_fun= lambda i, acc: acc - (log_probs * context_onehot[:,i,:]).sum(axis=1)  # minus since we need the NEG XE
        )
    
    loss = loss_per_sample.mean()
    return loss


"""
single sample version
"""
@jax.jit
def skipgram_softmax_forward_single_sample(params,  target):
    """Forward pass of the skipgram model.
    do the inner product of target and each word in the vocab

    V is the vocabulary size, D is the embedding dimension.
    params["projection"] is a (V, D) matrix of word embeddings.
    """
    # Indexing into (V, D) matrix with a batch of IDs. The output shape
    # is (batch_size, D).
    projection = params["projection"][target]  # dim,

    # all targets
    projection_target = params["projection"]   # dim, vocab

    hidden = jnp.einsum('d,vd->v', projection, projection_target)  # vocab
    return hidden

@jax.jit
def _skipgram_softmax_loss_single_sample(params, target, context):
    """Compute the loss of the word2Vec model."""
    
    """Compute the loss of the word2Vec model."""
    V, D = params['projection'].shape
    logits = skipgram_softmax_forward_single_sample(params, target)  #vocab,

    # those are the predictions given the center word
    # now check how those work against all different words in the context
    # softmax along the second dimension
    context_onehot = jax.nn.one_hot(context, V)  # (C, V)

    # calculate the loss for each context separately
    loss_per_context = jax.vmap(optax.losses.softmax_cross_entropy, in_axes=[None, 0]) (logits, context_onehot)
    return loss_per_context.sum() # sum over all context losses

@jax.jit
def skipgram_softmax_loss_single_sample(params, target_batch, context_batch):
    loss_per_sample = jax.vmap(_skipgram_softmax_loss_single_sample, in_axes=[None,0,0])
    return loss_per_sample(params, target_batch, context_batch).mean()
