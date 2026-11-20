# Classificação de imagens de fraturas ósseas

Projeto de visão computacional com exploração de imagens, classificação multirrótulo por transfer learning, avaliação, Grad-CAM e aplicação local em Streamlit.

**Protótipo educacional, sem validação clínica. Não utilizar para diagnóstico, triagem ou decisões de tratamento.**

## Abrir o projeto

- [Notebook para publicação](notebooks/01_fraturas.ipynb): métodos, gráficos agregados, métricas e código completo de análise.
- `notebooks/01_fraturas_local.ipynb`: versão local executada, incluindo grids de radiografias e Grad-CAM sobre as imagens.
- [Resultados](reports/resultados.md): resumo calculado da execução.
- [Auditoria do dataset](reports/auditoria.json): contagens, divisão e proveniência.

## Entregas

1. **Exploração:** validação das anotações YOLO, dimensões, distribuição das classes, coocorrência, grids por classe e divisão por grupos inferidos de origem.
2. **Modelo:** MobileNetV2 ImageNet congelada com cabeça Dense, regularização L2, dropout, aumento leve, redução da taxa de aprendizado e early stopping. Métricas multirrótulo, curvas de aprendizado, matrizes de confusão por classe e Grad-CAM.
3. **Aplicação:** upload local de JPG/PNG, escores por classe e mapa Grad-CAM. Treino e inferência separados em scripts.

## Por que multirrótulo?

O dataset contém **1.539 imagens**, das quais **113 têm mais de uma classe anotada**. Três não têm rótulo e são excluídas do treino supervisionado. O modelo retorna dez escores sigmoid independentes; ele não escolhe obrigatoriamente uma única classe.

As anotações originais são caixas YOLOv8. Para este projeto, o alvo da imagem é a união das classes anotadas. **Este modelo não detecta caixas nem localiza fraturas de forma validada.** Grad-CAM é um recurso de inspeção da rede, não um detector.

## Instalação

Ambiente validado: **Python 3.12, CPU no Windows**. As versões estão fixadas em `requirements.txt`; não se assume compatibilidade com Python 3.8.

Na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

No Linux/macOS, use `.venv/bin/python` nos comandos equivalentes. O treinamento inicial baixa os pesos ImageNet oficiais do Keras; requer internet. O modelo ajustado salvo em `models/fraturas.keras` funciona localmente após a instalação.

## Dados e reprodução

Fonte fornecida na atividade: [HBFMID — arquivo ZIP](https://cdn3.gnarususercontent.com.br/5327-checkpoint/Human%20Bone%20Fractures%20Multi-modal%20Image%20Dataset%20%28HBFMID%29.zip).
Fonte original indicada: [Human Bone Fractures Image Dataset](https://www.kaggle.com/datasets/jockeroika/human-bone-fractures-image-dataset).

O arquivo `data.yaml` identifica a licença como **Private**. Por isso, imagens, ZIP, grids e mapas sobre radiografias ficam fora do versionamento. A fonte é indicada em vez de republicar o dataset. Não é atribuída uma nova licença às imagens. O notebook público mantém o código das visualizações e os resultados agregados; as saídas de imagens locais são removidas antes da publicação.

```powershell
.\.venv\Scripts\python preparar_dados.py
.\.venv\Scripts\python treino.py --epochs 60
.\.venv\Scripts\python build_notebook.py
.\.venv\Scripts\python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_fraturas_local.ipynb --ExecutePreprocessor.timeout=600
.\.venv\Scripts\python finalizar_notebook.py
.\.venv\Scripts\python validate_project.py
```

O treinamento usa early stopping; 60 é o teto, não a quantidade obrigatória de épocas. Os vetores do backbone são calculados uma vez por execução para tornar o experimento viável em CPU. A reprodutibilidade numérica pode variar entre plataformas; a semente e as versões são registradas.

## Executar a aplicação

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Abra o endereço local mostrado no terminal. Carregue uma imagem JPG/PNG e, se desejar, visualize o Grad-CAM para uma classe. O aplicativo não salva a imagem nem chama serviços externos. Se a aplicação for hospedada em outro servidor, o upload será processado nesse servidor; este projeto não faz publicação automática da aplicação.

Para inferência sem interface:

```powershell
.\.venv\Scripts\python predicao.py --image "caminho/para/imagem.jpg"
```

## Divisão e prevenção de vazamento

- A origem fornece imagens aumentadas. **100 grupos de nomes de origem aparecem em mais de uma partição original.**
- A nova divisão mantém juntos nomes com o mesmo prefixo antes de `.rf.` e imagens de pixels idênticos, usando união transitiva.
- Foram utilizadas **1.536 imagens em 244 grupos inferidos**: 916 no treino, 328 na validação e 292 no teste.
- Nenhum grupo inferido ou hash exato de pixels cruza essas novas partições.
- Não há IDs de pacientes. Nomes podem colidir e diferentes nomes podem representar o mesmo paciente. O procedimento é conservador e **não garante separação por paciente ou remoção de todos os quase duplicados**.
- A escolha da partição usa somente rótulos e grupos; o teste não é usado para selecionar arquitetura, época ou limiar.

## Avaliação

Acurácia de subconjunto exige acerto de todos os rótulos da imagem. Precisão, recall e F1 macro dão o mesmo peso às classes; F1 micro agrega as decisões. As matrizes são 2×2 por classe, conforme a tarefa multirrótulo.

O limiar global é definido na validação. Os escores não são probabilidades clínicas calibradas, não somam necessariamente 100% e não permitem excluir fratura. A classe `Healthy` também pode ser prevista incorretamente. Os resultados completos estão em `reports/classificacao_test.csv` e `models/metadata.json`.

### Resultado desta execução

| Medida no teste | Valor |
| --- | ---: |
| Imagens / grupos inferidos | 292 / 49 |
| F1 macro | 0,219 |
| F1 micro | 0,301 |
| Precisão macro | 0,157 |
| Recall macro | 0,396 |
| Acurácia de subconjunto | 0,110 |

O early stopping restaurou a primeira época e interrompeu após oito. O treino apresenta desempenho superior à validação e ao teste, indicando dificuldade de generalização. `Linear` e `Segmental` tiveram F1 zero no teste. A entrega demonstra o pipeline e suas falhas; não apresenta o modelo como solução pronta.

## Estrutura

```text
app.py                         interface Streamlit
preparar_dados.py               download, auditoria e divisão agrupada
treino.py                       treinamento, regularização, limiar e métricas
predicao.py                     pré-processamento e inferência
src/modelo.py                   arquitetura e Grad-CAM
notebooks/                     análise pública e versão visual local
models/                        modelo .keras e metadados
data/manifest.csv               divisão, grupos e rótulos por imagem
data/annotations.csv            caixas originais em formato tabular
reports/figures/                gráficos agregados para publicação
reports/private_figures/        imagens e Grad-CAM locais, ignorados pelo Git
reports/                       auditoria, métricas e tabelas de avaliação
```

## Interpretabilidade e limitações

O Grad-CAM é calculado a partir dos mapas convolucionais finais (5×5 antes da ampliação). Pode destacar bordas, marcadores e texto, e não comprova localização correta de lesão. Os exemplos incluem casos com maior e menor F1 por imagem, sem ocultar falhas.

As classes raras possuem poucas imagens de origem; intervalos por grupo são exploratórios. Não há validação externa por paciente, calibração clínica ou avaliação prospectiva. A redução para 160×160 pode perder detalhes sutis e ImageNet tem domínio diferente das radiografias.

A inspeção dos exemplos encontrou imagens coloridas com aparência ilustrativa, além de imagens em escala de cinza. Não presumimos que toda imagem seja uma radiografia clínica autêntica. Essa heterogeneidade exige revisão da procedência e composição do dataset antes de qualquer outra conclusão.

## Referências de implementação

- [MobileNetV2 — Keras](https://keras.io/api/applications/mobilenet/).
- [Grad-CAM — exemplo oficial Keras](https://keras.io/examples/vision/grad_cam/).

## Revisão da entrega

O projeto inclui código, notebook, modelo, métricas e interface local. A aplicação não foi implantada em um serviço público e não deve ser utilizada com dados clínicos identificáveis sem as condições apropriadas de acesso e proteção.
