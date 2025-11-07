import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import heapq
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Suprime o FutureWarning do pandas

# --- CONFIGURAÇÃO E DADOS DE ENTRADA ---

# Ponto de Partida: Localização do Restaurante (Ponto Central)
RESTAURANTE_COORD = (-23.5505, -46.6333) # Exemplo: Centro de São Paulo
NUM_ENTREGADORES = 3 # K no K-Means
NUM_PEDIDOS = 25
VELOCIDADE_MEDIA_KMH = 30 # NOVIDADE: Velocidade média de um entregador em ambiente urbano

# 1. Geração de Dados de Pedidos Fictícios (Simulação)
np.random.seed(42)
pedidos_coords = np.array([
    RESTAURANTE_COORD[0] + np.random.uniform(-0.05, 0.05, NUM_PEDIDOS), # Latitude
    RESTAURANTE_COORD[1] + np.random.uniform(-0.05, 0.05, NUM_PEDIDOS)  # Longitude
]).T

df_pedidos = pd.DataFrame(pedidos_coords, columns=['Latitude', 'Longitude'])

# ID_Pedido = 0 para o Restaurante 
df_restaurante = pd.DataFrame([[0, RESTAURANTE_COORD[0], RESTAURANTE_COORD[1]]], columns=['ID_Pedido', 'Latitude', 'Longitude'])
df_pedidos.insert(0, 'ID_Pedido', range(1, NUM_PEDIDOS + 1))
df_pedidos = pd.concat([df_restaurante, df_pedidos], ignore_index=True)


print("--- 1. Dados de Pedidos Gerados (Amostra) ---")
# Renomeamos o ID 0 para 'Restaurante' apenas para visualização
df_temp = df_pedidos.copy()
df_temp['ID_Pedido'] = df_temp['ID_Pedido'].astype(str) 
df_temp.loc[df_temp['ID_Pedido'] == '0', 'ID_Pedido'] = 'Restaurante'
print(df_temp.head())
print("-" * 50)


# --- FASE 1: AGRUPAMENTO COM K-MEANS ---

def aplicar_kmeans(data, k):
    """Aplica o algoritmo K-Means para agrupar pedidos em K zonas."""
    # Para garantir a reprodutibilidade, usamos n_init=10 e random_state
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # K-Means nos dados dos pedidos (do índice 1 em diante)
    cluster_results = kmeans.fit_predict(data.iloc[1:][['Latitude', 'Longitude']])
    
    # Cria uma nova coluna 'Cluster'
    df_pedidos['Cluster'] = pd.Series(dtype=int)
    df_pedidos.loc[1:, 'Cluster'] = cluster_results
    df_pedidos.loc[0, 'Cluster'] = -1 # O Restaurante (ID 0) não pertence a um cluster de entrega
    
    print(f"--- 2. K-Means Aplicado (K={k} clusters) ---")
    print("Clusters de Pedidos:")
    # Remove o cluster do restaurante para contar
    print(df_pedidos[df_pedidos['Cluster'] != -1].groupby('Cluster').size())
    print("-" * 50)
    return df_pedidos


# --- FASE 2: OTIMIZAÇÃO DA ROTA COM ALGORITMO HEURÍSTICO (GREEDY) ---

def calcular_custo_simulado(coord1, coord2):
    """
    Calcula o custo (tempo de percurso em minutos) baseado na distância (aproximação).
    
    Usa a distância euclidiana simples convertida para KM (aproximando 1 grau ~= 111 km)
    e aplica a velocidade média definida para simular o tempo de viagem.
    """
    # Diferença de coordenadas
    dist_lat = coord2[0] - coord1[0]
    dist_lon = coord2[1] - coord1[1]
    
    # Distância Euclidiana em graus
    dist_euclidiana_graus = np.sqrt(dist_lat**2 + dist_lon**2)
    
    # Converte distância de graus para KM (Fator de ~111km/grau)
    dist_km = dist_euclidiana_graus * 111
    
    # Custo (Tempo) em minutos: (Distância em km / Velocidade em km/h) * 60
    custo_minutos = (dist_km / VELOCIDADE_MEDIA_KMH) * 60
    
    return custo_minutos

def resolver_rota_greedy_tsp(df_cluster, restaurante_coords):
    """
    Resolve a rota usando a Heurística do Vizinho Mais Próximo (Greedy).
    
    NOTA TÉCNICA: Este algoritmo é uma simplificação rápida do TSP (NP-Hard). 
    Ele encontra a melhor rota local a cada passo (o próximo ponto mais próximo), 
    mas não garante a rota globalmente ótima (o que um A* completo faria, mas seria 
    muito lento para este número de pontos).
    """
    pontos_visitar = df_cluster['ID_Pedido'].tolist()

    # Mapeia ID_Pedido para coordenadas
    coordenadas = {row['ID_Pedido']: (row['Latitude'], row['Longitude'])
                   for index, row in df_cluster.iterrows()}

    # O grafo de custo completo
    grafo_custo = {p: {} for p in pontos_visitar}
    for i in pontos_visitar:
        for j in pontos_visitar:
            if i != j:
                custo = calcular_custo_simulado(coordenadas[i], coordenadas[j])
                grafo_custo[i][j] = custo

    rota_otimizada = [0] # Começa no Restaurante (ID 0)
    pontos_pendentes = set([p for p in pontos_visitar if p != 0])
    custo_total = 0

    print("\n--- 3. Otimização da Rota (Algoritmo Heurístico Greedy - Vizinho Mais Próximo) ---")
    print(f"Iniciando no Ponto 0 (Restaurante). {len(pontos_pendentes)} pedidos para entregar.")

    # Loop para encontrar o próximo ponto mais próximo (Greedy)
    while pontos_pendentes:
        ponto_atual = rota_otimizada[-1]
        melhor_custo_transicao = float('inf')
        proximo_ponto = None

        # Busca pelo ponto mais próximo
        for proximo in pontos_pendentes:
            transicao_custo = grafo_custo[ponto_atual][proximo]

            if transicao_custo < melhor_custo_transicao:
                melhor_custo_transicao = transicao_custo
                proximo_ponto = proximo

        if proximo_ponto is not None:
            rota_otimizada.append(proximo_ponto)
            pontos_pendentes.remove(proximo_ponto)
            custo_total += melhor_custo_transicao
            print(f" -> Ponto {proximo_ponto} (Transição: {melhor_custo_transicao:.2f} min). Pendentes: {len(pontos_pendentes)}")
        else:
            print("Erro: Não foi possível encontrar um próximo ponto.")
            break
            
    # Retorna ao restaurante 
    custo_volta = calcular_custo_simulado(coordenadas[rota_otimizada[-1]], coordenadas[0])
    rota_otimizada.append(0)
    custo_total += custo_volta
    print(f" -> Ponto 0 (Restaurante, Volta: {custo_volta:.2f} min)")


    return rota_otimizada, custo_total


# --- FASE 3: VISUALIZAÇÃO DOS RESULTADOS (MAPA SIMULADO) ---

def plotar_rota_e_clusters(df_completo, rota_otimizada, cluster_escolhido):
    """Gera um gráfico de dispersão para visualizar clusters e a rota otimizada."""
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # 1. PLOTAR TODOS OS CLUSTERS (em cores diferentes)
    df_pedidos_sem_restaurante = df_completo[df_completo['ID_Pedido'] != 0]
    
    sns.scatterplot(
        x='Longitude', 
        y='Latitude', 
        hue='Cluster', 
        palette='deep', 
        data=df_pedidos_sem_restaurante, 
        s=100, 
        legend='full',
        zorder=2 
    )
    
    # 2. PLOTAR O RESTAURANTE (Ponto 0)
    plt.scatter(
        RESTAURANTE_COORD[1], 
        RESTAURANTE_COORD[0], 
        color='red', 
        marker='*', 
        s=300, 
        label='Restaurante (Ponto 0)',
        edgecolor='black',
        linewidth=1.5,
        zorder=3
    )

    # 3. PLOTAR A ROTA OTIMIZADA
    
    rota_coords = []
    for id_pedido in rota_otimizada:
        coord = df_completo[df_completo['ID_Pedido'] == id_pedido][['Latitude', 'Longitude']].iloc[0]
        rota_coords.append(coord)
    
    rota_df = pd.DataFrame(rota_coords, columns=['Latitude', 'Longitude'])

    # Traça as linhas da rota
    plt.plot(
        rota_df['Longitude'], 
        rota_df['Latitude'], 
        color='black', 
        linestyle='--', 
        linewidth=2, 
        alpha=0.6,
        label=f'Rota Heurística para Cluster {cluster_escolhido}', # Rótulo atualizado
        zorder=1 
    )

    # 4. ADICIONAR RÓTULOS (ID dos Pedidos) para a rota otimizada
    for i in range(len(rota_otimizada)):
        row = rota_df.iloc[i] 
        
        # Usa rótulo personalizado para o Restaurante (índice 0 e último)
        label_text = "Restaurante" if i == 0 or i == len(rota_otimizada) - 1 else f"P{rota_otimizada[i]}"
        
        # Ajusta a posição do rótulo para evitar sobreposição
        offset_y = -0.0005 if i % 2 == 0 else 0.0005
        
        plt.annotate(
            label_text, 
            (row['Longitude'], row['Latitude'] + offset_y), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center', 
            fontsize=9,
            fontweight='bold' if i == 0 else 'normal'
        )


    plt.title(f'Visualização da Rota Otimizada - Cluster {cluster_escolhido} (K-Means + Heurística Greedy)', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    
    # Reorganiza a legenda para evitar duplicidade
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles = {}
    for h, l in zip(handles, labels):
        if l not in unique_handles or 'Cluster' in l or 'Rota' in l:
            unique_handles[l] = h

    plt.legend(unique_handles.values(), unique_handles.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show() 


# --- EXECUÇÃO PRINCIPAL ---

# 1. Aplicar K-Means para agrupar
df_com_clusters = aplicar_kmeans(df_pedidos, NUM_ENTREGADORES)

# 2. Selecionar um Cluster para demonstração
cluster_escolhido = 0 
df_cluster_otimizar_pedidos = df_com_clusters[df_com_clusters['Cluster'] == cluster_escolhido].copy()
df_restaurante_row = df_com_clusters[df_com_clusters['ID_Pedido'] == 0].copy()
df_cluster_otimizar = pd.concat([df_restaurante_row, df_cluster_otimizar_pedidos], ignore_index=True)


# 3. Otimizar a Rota com o algoritmo Greedy
rota, custo = resolver_rota_greedy_tsp(df_cluster_otimizar, RESTAURANTE_COORD)

print("-" * 50)
print(f"*** ROTA FINAL PARA O CLUSTER {cluster_escolhido} (ENTREGADOR 1) ***")
print("Sequência de IDs: " + " -> ".join(map(str, rota)))
print(f"Custo Total Estimado da Rota (VELOCIDADE {VELOCIDADE_MEDIA_KMH} KM/H): {custo:.2f} minutos ({custo / 60:.2f} horas)")
print("Observação: Algoritmo Heurístico (Vizinho Mais Próximo) utilizado para otimizar o custo total (tempo).")
print("-" * 50)

# Demonstração do Resultado
rota_nomes = ["Restaurante" if i == 0 else f"Pedido {i}" for i in rota]
print("\nSequência de Visitas (Entregador 1):")
print(" -> ".join(rota_nomes))

# 4. PLOTAR O MAPA SIMULADO
plotar_rota_e_clusters(df_com_clusters, rota, cluster_escolhido)