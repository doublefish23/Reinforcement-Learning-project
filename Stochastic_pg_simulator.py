#Talha Sami
import gym

from stochastic_pg_agent import *

env = gym.envs.make("MountainCarContinuous-v0")

state_dim = env.observation_space.shape[0]

MAX_EPISODES = 1000
MAX_STEPS = 999


agent = TFRecurrentStochasticPolicyAgent2(env, num_input=1, init_learning_rate=5e-5, min_learning_rate=1e-10,
                                         learning_rate_N_max=2000, shuffle=True, batch_size=1)

render = True

if __name__ == "__main__":

    text_file = open("stochastic_pg.txt", "w")

    for episode_counter in range(MAX_EPISODES):
        state = env.reset()
        total_rewards = 0
        sigmas = []

        done = False

        step_counter=0

        for step_counter in range(MAX_STEPS):
            if render:
                env.render()
            action, sigma = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)

            total_rewards += reward
            sigmas.append(sigma)
            agent.store_rollout(state, action, reward)

            state = next_state
            if done:
                break

        agent.update_model(episode_counter)

        print("{},{:.2f},{}".format(episode_counter, total_rewards,step_counter))
        text_file.write("{},{:.2f},{}".format(episode_counter, total_rewards, step_counter))
        
    text_file.close()