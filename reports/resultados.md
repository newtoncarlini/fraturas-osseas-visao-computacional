# Resultados — classificação multirrótulo de fraturas

1539 imagens; 1536 utilizadas; 113 multirrótulo. 244 grupos inferidos; 100 cruzavam as partições originais.

## Modelo e avaliação
MobileNetV2 ImageNet congelada + Dense128 L2 + Dropout0.45 + sigmoid10. Limiar escolhido na validação: 0.40. 8 épocas executadas; melhor época por val_loss: 1.
- Teste: 292 imagens de 49 grupos inferidos.
- Subset accuracy: 0.1096.
- Precisão macro: 0.1572; recall macro: 0.3957.
- F1 macro: 0.2191; F1 micro: 0.3006.
- F1 macro da referência constante: 0.0422.

Acurácia significa coincidência exata do conjunto de rótulos, não acerto clínico de diagnóstico. Há classes raras com poucas origens e os grupos não são IDs de pacientes. Sem validação clínica; não utilizar para diagnóstico, triagem ou tratamento.
