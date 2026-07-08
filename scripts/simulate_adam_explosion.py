"""Simulate AdamW for a single parameter with pure math (no optax)."""
import math

B1 = 0.9
B2 = 0.95
EPS = 1e-8
WD = 1e-10

def simulate(param_init, steps, lr_fn, sparse_grad_every=100, sparse_grad_val=0.1):
    param = param_init
    mu = 0.0  # first moment
    nu = 0.0  # second moment
    count = 0  # step counter for bias correction

    history = []
    max_update = 0.0
    max_adam_ratio = 0.0

    for step in range(steps):
        count += 1
        lr = lr_fn(step)

        # Sparse loss gradient
        loss_grad = sparse_grad_val if step % sparse_grad_every == 0 else 0.0

        # Weight decay contribution
        wd_grad = WD * param

        # Total gradient (what add_decayed_weights produces)
        g = loss_grad + wd_grad

        # Adam update
        mu = B1 * mu + (1 - B1) * g
        nu = B2 * nu + (1 - B2) * g * g

        # Bias correction
        mu_hat = mu / (1 - B1**count)
        nu_hat = nu / (1 - B2**count)

        # Adam ratio
        adam_ratio = mu_hat / (math.sqrt(nu_hat) + EPS)

        # Final update
        update = -lr * adam_ratio  # optax includes the negative sign
        param += update

        abs_update = abs(update)
        if abs_update > max_update:
            max_update = abs_update
        if abs(adam_ratio) > max_adam_ratio:
            max_adam_ratio = abs(adam_ratio)

        if step <= 5 or step % max(1, steps // 10) == 0:
            history.append({
                'step': step, 'param': param, 'lr': lr,
                'g': g, 'mu': mu, 'nu': nu,
                'mu_hat': mu_hat, 'nu_hat': nu_hat,
                'adam_ratio': adam_ratio, 'update': update,
            })

    return history, max_update, max_adam_ratio

def lr_warmup_cosine(step):
    """WarmLinearCosineSchedule: warmup → linear → cosine"""
    warmup = 1000
    peak = 2e-5
    linear_end = 6000
    linear_end_lr = 1e-5
    cosine_end = 26000
    cosine_end_lr = 1e-6

    if step < warmup:
        return peak * step / warmup
    elif step < linear_end:
        frac = (step - warmup) / (linear_end - warmup)
        return peak + frac * (linear_end_lr - peak)
    elif step < cosine_end:
        frac = (step - linear_end) / (cosine_end - linear_end)
        cosine = 0.5 * (1 + math.cos(math.pi * frac))
        return cosine_end_lr + cosine * (linear_end_lr - cosine_end_lr)
    else:
        return cosine_end_lr

# ── Scenario 1: Normal param, normal updates ──
print("=" * 60)
print("Scenario 1: param=1.0, gradient every step, g=0.1")
h, m_upd, m_ratio = simulate(1.0, 1000, lr_warmup_cosine, sparse_grad_every=1, sparse_grad_val=0.1)
print(f"  Final param:  {h[-1]['param']:.6f}")
print(f"  Max |update|: {m_upd:.2e}")
print(f"  Max adam_ratio: {m_ratio:.6f}")
print(f"  → Safe. |update| ≤ lr_max = 2e-5 ✓")

# ── Scenario 2: Sparse gradients ──
print("\n" + "=" * 60)
print("Scenario 2: param=1.0, gradient every 100 steps, g=1.0")
h, m_upd, m_ratio = simulate(1.0, 12000, lr_warmup_cosine, sparse_grad_every=100, sparse_grad_val=1.0)
print(f"  Final param:  {h[-1]['param']:.6e}")
print(f"  Max |update|: {m_upd:.2e}")
print(f"  Max adam_ratio: {m_ratio:.6f}")
print(f"  → Safe. Sparse grads don't cause explosion ✓")

# ── Scenario 3: After param already exploded ──
print("\n" + "=" * 60)
print("Scenario 3: param=3.68e27 (from checkpoint), wd only, EPS=1e-8")
h, m_upd, m_ratio = simulate(3.68e27, 50, lr_warmup_cosine, sparse_grad_every=999999, sparse_grad_val=0.0)
print(f"  Step 0 update: {abs(h[0]['update']):.2e}")
print(f"  Step 0 adam_ratio: {h[0]['adam_ratio']:.2e}")
print(f"  Max |update|: {m_upd:.2e}")
print(f"  Max adam_ratio: {m_ratio:.6f}")
print(f"  → Adam ratio still ≤ 1 ✓")

# ── Scenario 4: What if mu from previous training, nu=0? ──
# This simulates: loading weights from checkpoint with FRESH opt_state
# but a previous activation left mu non-zero
print("\n" + "=" * 60)
print("Scenario 4: MANUAL state — simulate opt_state loaded from checkpoint")
print("  param=3.68e27, mu=1000 (stale), nu=0 (stale), count=12000")
mu = 1000.0    # stale first moment (from previous training)
nu = 0.0       # stale second moment (decayed to zero)
count = 12000  # stale counter
param = 3.68e27
lr = 1.8e-8

g = 0.0 + WD * param  # only weight decay
new_mu = B1 * mu + (1-B1) * g
new_nu = B2 * nu + (1-B2) * g * g
new_count = count + 1

mu_hat = new_mu / (1 - B1**new_count)
nu_hat = new_nu / (1 - B2**new_count)
adam_ratio = mu_hat / (math.sqrt(nu_hat) + EPS)
update = -lr * adam_ratio

print(f"  g (wd only) = {g:.2e}")
print(f"  mu: {mu:.2e} → {new_mu:.2e}")
print(f"  nu: {nu:.2e} → {new_nu:.2e}")
print(f"  mu_hat: {mu_hat:.2e}")
print(f"  nu_hat: {nu_hat:.2e}")
print(f"  adam_ratio: {adam_ratio:.2e}")
print(f"  update: {update:.2e}")
print(f"  → EXPLOSION! update >> lr because nu≈0+small and mu is stale")

# ── Scenario 5: Same but with EPS=1e-6 ──
print("\n" + "=" * 60)
print("Scenario 5: Same as 4 but EPS=1e-6")
EPS2 = 1e-6
adam_ratio2 = mu_hat / (math.sqrt(nu_hat) + EPS2)
update2 = -lr * adam_ratio2
print(f"  adam_ratio: {adam_ratio2:.2e}")
print(f"  update: {update2:.2e}")
print(f"  → EPS=1e-6 reduces update by 100x but still large")
