from io import BytesIO
import pandas as pd
import streamlit as st
from PIL import Image,UnidentifiedImageError
from predicao import load_artifacts,predict_image,overlay_gradcam

st.set_page_config(page_title='Explorador de fraturas',page_icon='🔬',layout='centered')
st.title('Explorador de imagens de fraturas')
st.warning('Demonstração educacional, sem validação clínica. Não use para diagnóstico, triagem ou decisões de tratamento.')
st.write('Carregue uma imagem para explorar os escores do modelo e o mapa de atenção.')

@st.cache_resource
def resources():
    return load_artifacts()

upload=st.file_uploader('Imagem JPG ou PNG',type=['jpg','jpeg','png'])
st.caption('O processamento ocorre na máquina que executa esta aplicação. A imagem não é salva pelo aplicativo.')
if upload is not None:
    if upload.size>20*1024*1024:
        st.error('Use um arquivo de até 20 MB.')
    else:
        try:
            image=Image.open(BytesIO(upload.getvalue()))
            image.verify()
            image=Image.open(BytesIO(upload.getvalue()))
            model,metadata=resources()
            test_metrics=metadata['metrics']['test']
            st.caption(f'Desempenho experimental no teste: F1 macro {test_metrics["f1_macro"]:.3f}; recall macro {test_metrics["recall_macro"]:.3f}. Há classes sem acertos no teste.')
            with st.spinner('Analisando a imagem…'):
                result,array=predict_image(image,model,metadata)
            table=pd.DataFrame(result).sort_values('escore',ascending=False)
            st.image(image,caption='Imagem enviada',width=400)
            selected=table.loc[table.acima_limiar,'classe'].tolist()
            if selected:
                st.write('**Rótulos acima do limiar:** '+', '.join(selected))
            else:
                st.info('Nenhuma classe ultrapassou o limiar. Isso não indica ausência de fratura.')
            if 'Healthy' in selected and len(selected)>1:
                st.info('A saída contém rótulos conflitantes. O modelo não impõe exclusividade entre classes.')
            st.caption(f'Limiar definido na validação: {metadata["threshold"]:.2f}. Os escores não são probabilidades clínicas calibradas e não somam necessariamente 100%.')
            st.dataframe(table.rename(columns={'classe':'Classe','escore':'Escore','acima_limiar':'Acima do limiar'}),hide_index=True)
            classe=st.selectbox('Classe para visualizar o Grad-CAM',metadata['classes'],index=metadata['classes'].index(table.iloc[0]['classe']))
            if st.button('Mostrar mapa de atenção'):
                with st.spinner('Gerando mapa…'):
                    overlay,_=overlay_gradcam(model,array,metadata['classes'].index(classe))
                st.image(overlay,caption=f'Grad-CAM — {classe}',width=400)
                st.caption('O mapa destaca regiões associadas ao escore, sem comprovar a localização de uma fratura ou a correção da previsão.')
            st.caption('O aplicativo não verifica se a imagem enviada é um raio-X nem detecta entradas fora do domínio de treino.')
        except (UnidentifiedImageError,ValueError,OSError) as error:
            st.error(f'Não foi possível processar a imagem: {error}')
