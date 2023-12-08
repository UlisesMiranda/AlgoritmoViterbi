import numpy as np

# Definición de las probabilidades de transición y emisión
prob_transicion = np.array([[0.7, 0.3], [0.4, 0.6]])  # Matriz de transición
prob_emision = np.array([[0.2, 0.8], [0.6, 0.4]])     # Matriz de emisión

# Observaciones
observaciones = ['U', 'N', 'U', 'N']
indice_observaciones = {'U': 0, 'N': 1}

# Nombres de estados
nombres_estados = ['Soleado', 'Lluvioso']

# Inicialización de probabilidades Viterbi en el día 1
prob_inicial = np.array([0.5, 0.5])  # Probabilidades iniciales
viterbi = np.zeros((len(observaciones), prob_transicion.shape[0]))

# Cálculo de las probabilidades Viterbi paso a paso
for t, obs in enumerate(observaciones):
    idx_obs = indice_observaciones[obs]
    if t == 0:
        viterbi[t] = prob_inicial * prob_emision[:, idx_obs]
    else:
        for i in range(prob_transicion.shape[0]):
            viterbi[t, i] = np.max(viterbi[t - 1] * prob_transicion[:, i]) * prob_emision[i, idx_obs]

# Secuencia de estados más probable
secuencia_estados = [np.argmax(viterbi[t]) for t in range(len(observaciones))]
probabilidad_resultante = np.max(viterbi[-1])

# Mostrar la secuencia de estados con su correspondiente clave
resultado = [(nombres_estados[estado], prob) for estado, prob in zip(secuencia_estados, np.max(viterbi, axis=1))]
print("\nSecuencia de estados con su clave (lluvioso/soleado) y probabilidad resultante:")
for idx, (estado, prob) in enumerate(resultado, start=1):
    print(f"Paso {idx}: {estado} ({prob:.6f})")

# Imprimir la probabilidad resultante
print(f"\nProbabilidad resultante: {probabilidad_resultante:.6f}")
