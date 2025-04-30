import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas as pd

# Función de entrenamiento


def train_q_learning(env, alpha, gamma, epsilon_max, epsilon_min, epsilon_decay, n_episodes=10000, max_steps=100, return_tracking=False):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards = []
    steps_per_episode = []
    epsilon_values = []

    epsilon = epsilon_max

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, truncated, info = env.step(action)

            # Actualizar Q-Table
            Q[state, action] += alpha * \
                (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        rewards.append(total_reward)
        steps_per_episode.append(step+1)  # porque empieza en 0
        epsilon_values.append(epsilon)

    if return_tracking:
        return Q, rewards, steps_per_episode, epsilon_values
    else:
        return Q


# Función de evaluación
def evaluate_q_learning(env, Q, n_tests=100, max_steps=100):
    successes = 0
    for _ in range(n_tests):
        state, _ = env.reset()
        for step in range(max_steps):
            action = np.argmax(Q[state, :])
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            if done:
                if reward == 1:
                    successes += 1
                break
    return successes / n_tests


# Función objetivo para Optuna
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    gamma = trial.suggest_float('gamma', 0.5, 0.99)
    epsilon_max = trial.suggest_float('epsilon_max', 0.8, 1.0)
    epsilon_min = trial.suggest_float('epsilon_min', 0.001, 0.1)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.9, 0.99999)

    env = gym.make("FrozenLake-v1", is_slippery=True)

    Q = train_q_learning(env, alpha, gamma, epsilon_max,
                         epsilon_min, epsilon_decay, n_episodes=10000)

    success_rate = evaluate_q_learning(env, Q, n_tests=100)

    env.close()

    return success_rate


# Crear y correr la optimización
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


# ---- GRAFICAR HISTORIA DE OPTIMIZACIÓN CON MATPLOTLIB ----

# Extraer datos
trials = study.trials
values = [trial.value for trial in trials]

plt.figure(figsize=(10, 6))
plt.plot(values, marker='o')
plt.xlabel("Número de intento (trial)")
plt.ylabel("Tasa de éxito")
plt.title("Historia de la Optimización (Optuna)")
plt.grid()
plt.show()


# Ahora entrena final con los mejores hiperparámetros
env = gym.make("FrozenLake-v1", is_slippery=True)

best_params = study.best_params

# Ahora sí, guardo rewards, steps y epsilon también
Q, rewards, steps_per_episode, epsilon_values = train_q_learning(
    env,
    alpha=best_params['alpha'],
    gamma=best_params['gamma'],
    epsilon_max=best_params['epsilon_max'],
    epsilon_min=best_params['epsilon_min'],
    epsilon_decay=best_params['epsilon_decay'],
    n_episodes=30000,
    return_tracking=True
)

# Evaluar política final
success_rate = evaluate_q_learning(env, Q, n_tests=100)
print(f"Tasa de éxito final: {success_rate*100:.2f}%")

env.close()

# Graficar rewards, pasos y epsilon
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Crear medias móviles
rewards_smooth = pd.Series(rewards).rolling(window=500, min_periods=1).mean()
steps_per_episode_smooth = pd.Series(steps_per_episode).rolling(
    window=500, min_periods=1).mean()

# Plot de rewards (media móvil)
axs[0].plot(rewards_smooth, color='blue')
axs[0].set_title("Rewards por episodio (media móvil 500)")
axs[0].set_xlabel("Episodio")
axs[0].set_ylabel("Reward")
axs[0].grid()

# Plot de pasos (sin media móvil)
axs[1].plot(steps_per_episode, color='green')
axs[1].set_title("Pasos por episodio")
axs[1].set_xlabel("Episodio")
axs[1].set_ylabel("Pasos")
axs[1].grid()

# Plot de epsilon (media móvil)
axs[2].plot(epsilon_values, color='red')
axs[2].set_title("Valor de epsilon por episodio (media móvil 500)")
axs[2].set_xlabel("Episodio")
axs[2].set_ylabel("Epsilon")
axs[2].grid()

plt.tight_layout()
plt.show()
