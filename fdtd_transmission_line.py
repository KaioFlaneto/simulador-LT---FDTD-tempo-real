import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# Configuração da Simulação
# ==========================================
# Parâmetros da Linha (Exemplo: Cabo Coaxial com perdas)
L = 250e-9       # Indutância [H/m]
C = 100e-12      # Capacitância [F/m]
R = 0.5          # Resistência [Ohm/m] (Aumente para ver mais atenuação)
G = 0.001        # Condutância [S/m]

# Parâmetros Geométricos e Temporais
length = 100.0       # Comprimento da linha [m]
Nz = 200             # Número de segmentos espaciais
dz = length / Nz     # Passo espacial [m]

# Cálculo de estabilidade (CFL Condition)
vp = 1 / np.sqrt(L * C)       # Velocidade de fase
Z0 = np.sqrt(L / C)           # Impedância característica aproximada
dt = 0.9 * dz / vp            # Passo temporal (90% do limite de Courant)
T_max = 5000                  # Número total de passos de tempo na animação

# Carga (Boundary Condition em z=L)
# Opções: 
# Z_L = 0 (Curto), Z_L = 1e9 (Aberto), Z_L = Z0 (Casado)
Z_L = 1e9 

print(f"--- Parâmetros da Linha ---")
print(f"Z0 (Impedância Característica) : {Z0:.2f} Ohms")
print(f"Velocidade de Fase             : {vp/1e8:.2f} x 10^8 m/s")
print(f"Passo Espacial (dz)            : {dz:.4f} m")
print(f"Passo Temporal (dt)            : {dt*1e9:.2f} ns")
print(f"Carga (ZL)                     : {Z_L:.2e} Ohms")

# ==========================================
# Inicialização dos Arrays (FDTD)
# ==========================================
# V[k] representa tensão na posição k
# I[k] representa corrente na posição k+0.5
V = np.zeros(Nz + 1)
I = np.zeros(Nz)

# Fonte de Tensão (Pulso Gaussiano)
def source_voltage(t_step):
    # Parâmetros do pulso
    t0 = 200  # Centro do pulso (em steps)
    sigma = 50 # Largura do pulso
    return 10.0 * np.exp(-((t_step - t0)**2) / (2 * sigma**2))

# Coeficientes para atualização FDTD (Linha com Perdas)
# Baseado na discretização semi-implícita das Eq. dos Telegrafistas
Cv1 = (2*C - G*dt) / (2*C + G*dt)
Cv2 = (2*dt) / ((2*C + G*dt) * dz)

Ci1 = (2*L - R*dt) / (2*L + R*dt)
Ci2 = (2*dt) / ((2*L + R*dt) * dz)

# ==========================================
# Configuração da Visualização (Matplotlib)
# ==========================================
plt.style.use('dark_background') # Estilo "Cyberpunk" / Científico
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle(f'Simulação FDTD de Linha de Transmissão\nZ0={Z0:.1f}$\\Omega$, ZL={Z_L:.1e}$\\Omega$', fontsize=16, color='cyan')

# Plot da Tensão
line_v, = ax1.plot([], [], color='#00ffcc', lw=2, label='Tensão (V)')
ax1.set_ylabel('Tensão [V]', fontsize=12)
ax1.set_ylim(-15, 15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')

# Plot da Corrente
line_i, = ax2.plot([], [], color='#ff00cc', lw=2, label='Corrente (I)')
ax2.set_ylabel('Corrente [A]', fontsize=12)
ax2.set_xlabel('Posição [m]', fontsize=12)
ax2.set_ylim(-15/Z0*2, 15/Z0*2) # Escala automática baseada na Z0
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right')

# Texto de informações
time_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, color='white', fontsize=10)

z_axis = np.linspace(0, length, Nz + 1)
z_axis_i = np.linspace(0, length, Nz) # Grid levemente deslocado para corrente

# ==========================================
# Função de Atualização (Loop de Simulação)
# ==========================================
def update(n):
    global V, I
    
    # 1. Atualiza Tensão (V)
    # V[1:Nz] depende de I[0:Nz-1] e I[1:Nz]
    # Equação: dV/dz = ... -> V(k) é afetado por I(k) e I(k-1)
    # Na malha de Yee 1D: V_k está entre I_k-1 e I_k
    
    # V_new[k] = Cv1 * V_old[k] - Cv2 * (I_old[k] - I_old[k-1])
    V[1:Nz] = Cv1 * V[1:Nz] - Cv2 * (I[1:Nz] - I[0:Nz-1])
    
    # Condição de Contorno na Fonte (Hard Source)
    V[0] = source_voltage(n)
    
    # Condição de Contorno na Carga (Z_L)
    # Modelo simples para resistiva: V_N = Z_L * I_N_aprox
    # Para FDTD preciso, usamos a reflexão local aproximada ou I fantasma.
    # Implementação simples: Equação de contorno discreta
    if Z_L > 1e8: # Circuito Aberto (aprox)
        # I[Nz-1] = 0 (na verdade I define o fluxo para fora, I=0 no final)
        # Mas V[Nz] precisa ser atualizado.
        # Em aberto, I na ponta é 0.
        # V[Nz] = V[Nz] - dt/C * (0 - I[Nz-1])/dz ??
        # Aprox de 1a ordem: V[Nz] replica V[Nz-1] (reflexão total) ou atualiza com I=0
        V[Nz] = Cv1 * V[Nz] - Cv2 * (0 - I[Nz-1])
    elif Z_L < 1e-3: # Curto Circuito
        V[Nz] = 0.0
    else:
        # Carga Resistiva Casada ou genérica
        # I_load = V[Nz] / Z_L
        # Atualização explícita requer cuidado. 
        # Usaremos uma aproximação onde a corrente na carga é calculada com o V atual
        I_load = V[Nz] / Z_L
        V[Nz] = Cv1 * V[Nz] - Cv2 * (I_load - I[Nz-1])

    # 2. Atualiza Corrente (I)
    # I_new[k] = Ci1 * I_old[k] - Ci2 * (V_new[k+1] - V_new[k])
    I[0:Nz] = Ci1 * I[0:Nz] - Ci2 * (V[1:Nz+1] - V[0:Nz])
    
    # Atualiza gráficos
    line_v.set_data(z_axis, V)
    line_i.set_data(z_axis_i, I)
    
    time_text.set_text(f'Tempo: {n*dt*1e9:.1f} ns')
    
    return line_v, line_i, time_text

# Inicializa gráficos
ax1.set_xlim(0, length)
ax2.set_xlim(0, length)

# Cria animação
# interval=20ms -> 50fps
anim = FuncAnimation(fig, update, frames=T_max, interval=1, blit=True)

print("Iniciando animação...")
plt.show()
