import ppo

if __name__ == '__main__':
    net = ppo.CPPO()
    net.train()
    net.rendering()