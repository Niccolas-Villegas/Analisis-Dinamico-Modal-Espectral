from P2_Analisis_Dinamico import *
from units import *

# Datos de entrada
# ----------------
w = [60, 61, 61, 52]                  # Peso de entrepiso (tnf)
k = [47600, 24780, 23450, 21230]      # Rigidez de entrepiso (tnf/m)
h = [2.5, 3, 3, 3]                    # Altura de entrepiso (m)
g = 9.81                              # Aceleracion de la gravedad (m/s2)

# Asignacion de unidades MKS
# --------------------------
for i in range(len(w)):
    w[i] = w[i]*(tnf/g)
    k[i] = k[i]*(tnf/m)
    h[i] = h[i]*m


# Parametros Sismicos (Norma E.030)
# ---------------------------------
Z = 0.45               # Factor de zonificacion
U = 1.00               # Factor de uso de importancia
S = 1.05               # Factor de suelo
R = 6.00               # Coeficiente de Reduccion de fuerzas sismicas
Tp = 0.6               # Periodo inicial del tramo constante del factor C
Tl = 2.0               # Periodo final del tramo constante del factor C

S_coeff = Z*U*S*g/R    # Coeficiente sismico



# Formas de modo (graficos)
# -------------------------
x1 = Analisis_Modal(w, k, h)
x1.Graficos()

# Formas de modo (valores)
# ------------------------
print(x1.modes)

# Respuestas máximas
# ------------------
print(x1.results)


# Desplazamiento, derivas y cortantes
# -----------------------------------
x2 = Analisis_Espectral(w, k, h, S_coeff, R, Tp, Tl)
x2.Graficos()