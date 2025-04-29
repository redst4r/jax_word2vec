import jax.numpy as jnp
import jax
from functools import partial

class Skipgram_NegSam():
    def __init__(self):
        pass
        
    @partial(jax.jit, static_argnums=(0,))
    def _positive_loss(self, params, target, context):
        """The loss incurred by positive samples (i.e. context"""
        
        context_rep = params["projection"][context]     #(batchsize, C,  D)
        target_rep = params["projection"][target]   #(batchsize, D)
    
        # prediction for each context
        logits_pos = jnp.einsum('bcd,bd->bc', context_rep, target_rep)  # batch, C
        pos_loss_per_Sample = -jax.nn.log_sigmoid(logits_pos).sum(axis=1)  # summing the loss over all contexts per sample
        pos_loss = pos_loss_per_Sample.mean()    
        
        return pos_loss

    @partial(jax.jit, static_argnums=(0,))
    def _negative_loss(self, params, target, neg_samples):
        """The loss incurred by negative samples"""
        
        target_rep = params["projection"][target]   #(batchsize, D)
        # prediction for each negative sample
        negative_target_embeddings = params["projection"][neg_samples]  # batch x n_neg x dim
        negative_logits = jnp.einsum("bd,bnd->bn", target_rep, negative_target_embeddings)   # batch x n_neg
        neg_loss_per_Sample = -jax.nn.log_sigmoid(-negative_logits).sum(axis=1)  # mind the -1 in the sigmoid! also we sum over all neg samples
        neg_loss = neg_loss_per_Sample.mean()
        return neg_loss
        
        
    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, target, neg_samples, context):
        pos_loss = self._positive_loss(params, target=target, context=context)
        neg_loss = self._negative_loss(params, target=target, neg_samples=neg_samples)
        return pos_loss + neg_loss


"""
the non-batched versions
"""
class Skipgram_NegSam_single():
    def __init__(self):
        pass
        
    @partial(jax.jit, static_argnums=(0,))
    def _positive_loss(self, params, target, context):
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
        return pos_loss_per_Sample

    @partial(jax.jit, static_argnums=(0,))
    def _negative_loss(self, params, target, neg_samples):
        """
        dimensionas:
        C: number of context
        D: latent dim
        N: number of neg examples
        
        target: shape: int
        context: shape: int[C]
        neg_samples: shape: int[Neg]
        """
        target_rep = params["projection"][target]    # (D,)
        negative_target_embeddings = params["projection"][neg_samples]  # n_neg x dim
        negative_logits = jnp.einsum("d,nd->n", target_rep, negative_target_embeddings)   # batch x n_neg
        neg_loss_per_Sample = -jax.nn.log_sigmoid(-negative_logits).sum()  # mind the -1 in the sigmoid! also we sum over all neg samples
    
        return neg_loss_per_Sample
    
    @partial(jax.jit, static_argnums=(0,))
    def _loss_single(self,  params, target, neg_samples, context):
        p = self._positive_loss(params, target=target, context=context)
        n = self._negative_loss(params, target=target, neg_samples=neg_samples)
        return n+p
        
    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params, target_batch, neg_samples_batch, context_batch):
        """
        feed in batches, but uses vmap to calculate acrsoss batches
        """
       
        f_batch = jax.vmap(self._loss_single, in_axes=[None, 0,0,0])
        return f_batch(params, target_batch, neg_samples_batch, context_batch).mean()
    
    





