# RotaInteligente Otimiza o de Entregas com Algoritmos de IA
Rota Inteligente: Otimização de Entregas com Algoritmos de IA para  empresa Sabor Express


**Foco Principal: Maximização da Eficiência Logística (Busca em Grafos)**

Este projeto de Inteligência Artificial visa solucionar o problema de ineficiência logística enfrentado pela empresa de delivery Sabor Express durante os horários de pico. O objetivo é substituir o planejamento de rotas manual e ineficaz por uma solução inteligente, baseada em algoritmos de busca em grafos, que determina o menor caminho e a sequência de entregas mais rápida entre múltiplos pontos.

A iniciativa transforma o mapa da cidade em um Grafo (com locais de entrega sendo os nós e as ruas, com seus tempos de viagem, sendo as arestas) para calcular, em tempo real, a rota ideal, otimizando custos e elevando a satisfação do cliente.



**Etapa 1 – Geração do Cenário e Estrutura de Dados**
O projeto utiliza a simulação de um cenário urbano para modelar o problema. O código gera e processa um grafo que representa as conexões e o custo (tempo de viagem) entre diferentes pontos da cidade.

**Estrutura de Dados Chave:**
*  Nós (Nodes): Representam os locais de entrega (pedidos) e o ponto de origem (restaurante).
*  Arestas (Edges): Representam as ruas ou conexões entre os nós. O Peso da Aresta (custo) é definido pelo tempo estimado de viagem.
*  Variável de Otimização: O custo total da rota (soma dos pesos das arestas) deve ser minimizado.

  

**Parâmetros da Simulação:**

*  Geração de N pontos de entrega aleatórios com pesos simulados.
*  Definição do ponto de partida fixo.



**Etapa 2 – Aplicação do Algoritmo de Inteligência Artificial**
A otimização da rota é realizada através de um algoritmo heurístico de busca, que garante que a solução encontrada seja não apenas viável, mas a mais eficiente possível (menor custo).


**Algoritmo Aplicado:**
*  Algoritmo A* (A-Star): Escolhido por ser um algoritmo de busca informada altamente eficiente, que combina o custo já percorrido (g(n)) com uma estimativa heurística (h(n)) do custo restante para o objetivo. Isso permite encontrar o caminho mais curto de forma rápida, ideal para ambientes dinâmicos de delivery.


**Principais Análises e Resultados Gerados:**
*  Cálculo de Eficiência: Comparação direta do Custo Total (tempo/distância) da rota gerada pela IA versus uma rota manual (simulada ou de linha reta).
*  Visualização da Solução: Demonstração da rota otimizada sobre o grafo, destacando a sequência de paradas sugerida pelo algoritmo.



**Saídas de Visualização:**
*  grafico_grafo_otimizado.png (Visualização dos nós, arestas e a rota final em destaque).
*  tabela_comparacao_custos.txt (Comparação do tempo/distância total entre a Rota Manual vs. Rota IA).

  

**Tecnologias Utilizadas**

*  Python 3: Linguagem principal para desenvolvimento.
*  NumPy / Pandas: Estruturação e manipulação eficiente dos dados do grafo e das distâncias/custos.
*  NetworkX (Opcional, ou Estruturas Próprias): Para modelagem e manipulação do grafo.
*  Matplotlib: Geração das visualizações do grafo e da rota.




**Resultados e Conclusão

*  O projeto entrega uma solução prática para a Sabor Express, demonstrando a superioridade da IA no planejamento logístico em comparação com métodos manuais. O modelo gera um conhecimento prático para:
*  Redução de Custos: Diminuição significativa no consumo de combustível e menor desgaste dos veículos ao evitar caminhos longos e ineficientes.
*  Melhoria da Qualidade de Serviço: Aceleração do tempo de entrega, garantindo a satisfação e fidelidade do cliente (alimentos quentes, previsões precisas).
*  Escalabilidade Operacional: Capacidade de processar grandes volumes de pedidos rapidamente, permitindo à empresa crescer sem comprometer a eficiência.


---
Autor: Igor Ferreira Alves

Curso: Engenharia da Computação — UNIFECAF (2025)
