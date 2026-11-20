"""Gera cópia pública sem incorporar as radiografias e seus mapas sobrepostos."""
from pathlib import Path
import json
import nbformat

root=Path(__file__).resolve().parent
nb=nbformat.read(root/'notebooks/01_fraturas_local.ipynb',as_version=4)
meta=json.loads((root/'models/metadata.json').read_text(encoding='utf-8'))
t=meta['metrics']['test']
old='Os resultados calculados aparecem na avaliação e em `reports/resultados.md`.'
new=(f'**1.536 imagens rotuladas**, dez classes e 244 grupos de origem inferidos. '
     f'No teste separado: **F1 macro {t["f1_macro"]:.3f}**, **recall macro {t["recall_macro"]:.3f}** e '
     f'**acurácia de subconjunto {t["subset_accuracy"]:.3f}**. Os resultados detalhados e limitações aparecem abaixo. '
     'Este desempenho experimental não valida uso clínico.')
nb.cells[0].source=nb.cells[0].source.replace(old,new)
nbformat.write(nb,root/'notebooks/01_fraturas_local.ipynb')
count=0
for cell in nb.cells:
    if cell.cell_type=='code' and 'private-images' in cell.metadata.get('tags',[]):
        count+=len(cell.outputs); cell.outputs=[]
nbformat.validate(nb)
nbformat.write(nb,root/'notebooks/01_fraturas.ipynb')
print(f'Cópia pública criada; {count} saídas locais removidas. Código e métricas preservados.')
