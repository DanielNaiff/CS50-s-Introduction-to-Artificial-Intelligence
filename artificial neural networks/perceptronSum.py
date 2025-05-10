import random
import matplotlib.pyplot as plt

# Gera os dados de treino
train_data = [(x1, x2, x1 + x2) for x1 in range(50) for x2 in range(50)]

# Inicializa pesos e bias
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = 0.0

learning_rate = 0.0001
epochs = 1000
errors = []

# Treinamento
for epoch in range(epochs):
    total_error = 0
    for x1, x2, y_true in train_data:
        # Forward
        y_pred = w1 * x1 + w2 * x2 + b

        # Erro
        error = y_true - y_pred
        total_error += error ** 2

        # Gradientes
        dw1 = -2 * x1 * error
        dw2 = -2 * x2 * error
        db = -2 * error

        # Atualiza os parâmetros
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b  -= learning_rate * db

    errors.append(total_error)

# Resultados finais
print(f"w1: {w1:.4f}, w2: {w2:.4f}, b: {b:.4f}")
print(f"Previsão de 20 + 30 = {w1 * 20 + w2 * 30 + b:.2f}")

# Plot do erro
plt.plot(errors)
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro quadrático total")
plt.grid(True)
plt.show()
