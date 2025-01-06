import ppo

if __name__ == '__main__':
    net = ppo.CPPO()
    net.train()
    while True:
        import config
        import torch
        with torch.no_grad():
            net.env.rollout(
                max_steps=1000,
                policy=net.policy,
                callback=lambda env, _: env.render(),
                auto_cast_to_device=True,
                break_when_any_done=False,
            )