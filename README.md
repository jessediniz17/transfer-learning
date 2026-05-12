# Classificação de Imagens de Gatos e Cachorros com Transfer Learning
# Visão Geral do Projeto
Este projeto demonstra o uso de Transfer Learning para construir um modelo de classificação de imagens capaz de distinguir entre gatos e cachorros. Utilizamos o modelo pré-treinado MobileNetV2 como extrator de características e adicionamos camadas de classificação personalizadas para a tarefa específica. O objetivo é alcançar alta precisão na classificação de imagens, aproveitando o conhecimento adquirido por um modelo em um grande conjunto de dados de imagens.

# Conjunto de Dados
O projeto utiliza o conjunto de dados cats_vs_dogs, acessado através da biblioteca tensorflow_datasets (tfds). O conjunto de dados foi dividido da seguinte forma:

Treinamento: 80% do conjunto de dados original
Validação: 10% do conjunto de dados original
Teste: 10% do conjunto de dados original
Pré-processamento de Dados
As imagens foram pré-processadas para atender aos requisitos de entrada do MobileNetV2:

# Redimensionamento: Todas as imagens foram redimensionadas para (224, 224) pixels.
Normalização: Os valores dos pixels foram convertidos para float32 e normalizados para o intervalo [0, 1].
Batching e Prefetching: Os dados foram agrupados em batches de 32 imagens e otimizados para desempenho de carregamento com tf.data.AUTOTUNE.
Arquitetura do Modelo
A arquitetura do modelo é baseada no conceito de Transfer Learning:

# Modelo Base MobileNetV2:

Utilizamos o MobileNetV2 pré-treinado no conjunto de dados ImageNet.
As camadas convolucionais do MobileNetV2 (excluindo as camadas classificadoras do topo) são usadas como um extrator de características.
Os pesos do modelo base foram congelados (base_model.trainable = False) para evitar que sejam atualizados durante o treinamento inicial, preservando as características aprendidas.
Camadas de Classificação Personalizadas:

Uma camada GlobalAveragePooling2D é adicionada após o modelo base para reduzir as dimensões espaciais.
Uma camada Dense com 1 neurônio e função de ativação sigmoid é usada como a camada de saída para classificação binária (gato ou cachorro).
Treinamento do Modelo
O modelo foi compilado e treinado com as seguintes configurações:

# Otimizador: Adam (tf.keras.optimizers.Adam())
Função de Perda: BinaryCrossentropy (tf.keras.losses.BinaryCrossentropy()), apropriada para classificação binária.
Métricas: Acurácia (accuracy).
Épocas: O modelo foi treinado por 10 épocas.
Avaliação de Desempenho
Após o treinamento, o modelo foi avaliado no conjunto de dados de teste.

Perda no Teste: 0.0402
Acurácia no Teste: 98.62%
# Resultados e Conclusão
O modelo alcançou uma acurácia final de teste de 98,62% com uma perda de teste de 0,0402, demonstrando um excelente desempenho em dados não vistos. As visualizações do histórico de treinamento indicaram que tanto a acurácia de treinamento quanto a de validação aumentaram consistentemente ao longo das 10 épocas, enquanto as perdas de treinamento e validação diminuíram. A acurácia de treinamento atingiu mais de 99%, e a acurácia de validação estabilizou em torno de 98,45%.

A proximidade entre as métricas de treinamento e validação sugere que o modelo não sofreu de sobreajuste significativo, aprendendo características generalizáveis. A alta e estável acurácia de validação reforça a robustez do modelo. Em resumo, a abordagem de Transfer Learning com MobileNetV2 foi altamente eficaz para a classificação de gatos e cachorros.

# Como Executar
Certifique-se de ter um ambiente Python com TensorFlow e TensorFlow Datasets instalados.
Execute as células do notebook em ordem.
O processo inclui o download do conjunto de dados, pré-processamento, construção do modelo, treinamento e avaliação.
As visualizações geradas no notebook fornecem insights sobre o progresso do treinamento.

# Recursos
TensorFlow Documentation
Keras Applications
TensorFlow Datasets
MobileNetV2: Inverted Residuals and Linear Bottlenecks
