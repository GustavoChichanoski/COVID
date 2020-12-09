# Classificação de doenças

A intenção de projeto é alterar o artigo cientifico [10.1109/TMI.2020.2993291](https://ieeexplore.ieee.org/document/9090149) para utilizar usando o framework keras.

## Características do projeto

### Linguagens

- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/768px-Python-logo-notext.svg.png" alt="python" style="zoom:2%" /> Python

## Datasets

- [Segmentação](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)
- [Classificação](https://www.kaggle.com/andrewmvd/covid19-ct-scans)

## Continuidade do projeto

- [x] **Segmentação do pulmão**
  - [x] Leitura do _dataset_ 
    - [x] Leitura dos pulmões
      - [x] Leitura da imagem com **OpenCV**
      - [x] Redimensionamento da imagem
      - [x] Conversão em escala de cinza
      - [x] Normalização do histograma
      - [x] Ajuste da gama da Imagem
      - [x] Reescalonamento da Imagem de -1 a 1
    - [x] Leitura da mascara
      - [x] Leitura da imagem com **OpenCV**
      - [x] Redimensionamento da imagem
      - [x] Reescalonamento da Imagem de -1 a 1
  - [x] Criação do modelo do **_Deep Learning_**
    - [x] Criação do modelo U-Net
    - [x] Testes com diversos parâmetros para comparação de resultados
    - [x] Treinamento do modelo
    - [x] Escolha do modelo
  - [x] Resultados
    - [x] Apresentar as métricas de avaliação
    - [x] Apresentar as predições com o _dataset_ de treino
    - [x] Comparar as mascaras obtidas com as mascaras reais
- [ ] **Classificação das Imagens**
  - [x] Leitura do _dataset_
    - [x] Leitura da imagem segmentada
    - [x] Dividir a imagens em partes
    - [x] Reescalonamento da Imagem de -1 a 1
  - [x] Criação do modelo
    - [x] Importação do ResNet50V2
    - [x] Criação das camadas de _Full Connected_
    - [x] Compilação do modelo
    - [ ] Treinamento
  - [ ] Resultados
    - [x] Aplicar Grad-Cam Probabilistico
      - [x] Aplicar Grad-Cam para uma unica imagem.
      - [x] Aplicar Grad-Cam em multiplas imagens.
      - [x] Juntar as imagens
