import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Crear el environment
env = gym.make("FrozenLake-v1", is_slippery=True)  # mapa 4x4

# Inicializar Q-Table
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

# Hiperparámetros
alpha = 0.3
gamma = 0.995
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9998
epsilon_max = 1.0

n_episodes = 50000
max_steps = 100

# Variables para seguimiento
rewards = []
steps_per_episode = []
epsilons = []
best_avg_reward = 0
best_Q = Q.copy()

# Entrenamiento
for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    for _ in range(max_steps):
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
        steps += 1

        if done:
            break

    # Decaer epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Guardar métricas
    rewards.append(total_reward)
    steps_per_episode.append(steps)
    epsilons.append(epsilon)

    # Mostrar cada 500 episodios y verificar si es la mejor Q-table
    if (episode + 1) % 500 == 0:
        promedio_reward = np.mean(rewards[-500:])
        promedio_step = np.mean(steps_per_episode[-500:])
        print(
            f"Episodio {episode+1}, Reward promedio ultimos 500 episodios: {promedio_reward:.2f}, Epsilon: {epsilon:.3f}, Step por Episodio: {promedio_step:.2f}")

        if promedio_reward > best_avg_reward:
            best_avg_reward = promedio_reward
            best_Q = Q.copy()
            print("Nueva mejor Q-table guardada.")

# Guardar la mejor Q-table
np.save("mejor_Q_table.npy", best_Q)

# Graficar recompensas


def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Crear el gráfico con los tres subgráficos
plt.figure(figsize=(18, 6))

# Gráfico 1: Reward promedio
plt.subplot(1, 3, 1)
plt.plot(moving_average(rewards))
plt.title(f'Reward promedio)')
plt.xlabel('Episodios')
plt.ylabel('Reward')
plt.grid()

# Gráfico 2: Pasos por episodio
plt.subplot(1, 3, 2)
plt.plot(moving_average(steps_per_episode))
plt.title(f'Pasos por episodio)')
plt.xlabel('Episodios')
plt.ylabel('Pasos')
plt.grid()

# Gráfico 3: Decaimiento de epsilon
plt.subplot(1, 3, 3)
plt.plot(epsilons)
plt.title('Decaimiento de epsilon')
plt.xlabel('Episodios')
plt.ylabel('Epsilon')
plt.grid()

# Ajustar los gráficos y mostrarlos
plt.tight_layout()
plt.show()

# Evaluar desempeño con la mejor Q-table (sin exploración)
Q = best_Q
n_test = 100
successes = 0

for _ in range(n_test):
    state, _ = env.reset()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        # env_test.render()  # Mostrar el tablero con el muñeco moviéndose
        if done:
            if reward == 1:  # Si el reward es 1, significa que llegó al objetivo
                successes += 1
            break

print(f"Éxito en {successes}/{n_test} pruebas.")
