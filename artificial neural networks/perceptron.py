import matplotlib.pyplot as plt

# Dados de treino (Celsius → Fahrenheit)
celsius = [0, 10, 20, 30, 40, 50, 60]
fahrenheit = [32, 50, 68, 86, 104, 122, 140]

# Inicializa peso e bias
w = 0.0  # peso
b = 0.0  # bias
learning_rate = 0.0005
epochs = 10000

# Guarda os erros ao longo do tempo
errors = []

# Treinamento
for epoch in range(epochs):
    total_error = 0
    for x, y_true in zip(celsius, fahrenheit):
        # Forward: predição
        y_pred = w * x + b
        
        # Erro
        error = y_true - y_pred
        total_error += error ** 2
        
        # Gradientes
        dw = -2 * x * error
        db = -2 * error

        # Atualiza pesos
        w = w - learning_rate * dw
        b = b - learning_rate * db

    errors.append(total_error)

# Resultados finais
print(f"Peso final (w): {w:.4f}")
print(f"Bias final (b): {b:.4f}")

# Testando o modelo
def predict(c):
    return w * c + b

print(f"Previsão para 100°C: {predict(100):.2f} °F (esperado: 212 °F)")

# Plot do erro
plt.plot(errors)
plt.title("Erro ao longo das épocas")
plt.xlabel("Épocas")
plt.ylabel("Erro quadrático total")
plt.grid(True)
plt.show()
