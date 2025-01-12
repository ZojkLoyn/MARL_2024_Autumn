import config
import ppo
import time

if __name__ == '__main__':
    ppo = ppo.get_ppo(critic_net_load=config.load, policy_net_load=config.load)
    if config.render:
        total_rewards = []
    for ep in range(config.train_epochs):
        if config.train:
            ppo.train()
            ppo.save()
        if config.render:
            rewards = ppo.render(record=True, record_path=f"videos/{time.time()}.mp4")
            total_rewards.append(rewards)
    
    if config.render:
        print(total_rewards)