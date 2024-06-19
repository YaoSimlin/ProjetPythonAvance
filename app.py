import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import ydata_profiling

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data


def main():
    st.title("YaoML")
    st.sidebar.write("[Auteur: Yao Jean Bosco]")
    st.sidebar.markdown(
        "** Cette application est une application No-Code pour l'exploration, l'analyse de données**" )
    
    file =st.file_uploader("charger votre fichier csv ici:",type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head())

    profile = st.button("Rapport Dataset")
    if profile:
        profile_df = data.profile_report()
        st_profile_report(profile_df)

    VarCible = st.selectbox("variable cible ",data.columns)
    tache = st.selectbox("Choisir action",["Regression","Classification"])
    if st.button("Executer"):
        exo_reg = setup_reg(data, target=tache)
        model_reg = compare_models_reg()
        save_model_reg(model_reg,"meilleur modèle")
        st.success("Le modèle de regression a été construit avec succès!")

        #Affichage des resultats
        if tache == "Regression":
            st.write("Residus")
            plot_model_reg(model_reg,plot= 'residuals', save=True)
            st.image("Residus.png")

            st.write("Feature Importance")
            plot_model_reg(model_reg,plot= 'feature', save= True)
            st.image("Feature Importance.png")

            with open('meilleur modèle.pk1','rb') as f:
                st.download_button('telecharger le Pipeline modèle',f,file_name="meilleur modèle.pk1")

        if tache == "Classification":
            if st.button("Executer"):
                exp_class = setup_class(data,target= tache)
                model_class = compare_models_class()
                save_model_class(model_class,'meilleur modèle')
                st.success("modèle de classification exécuté avec succès!")
                #affichage des resultats
                col5,col6 = st.columns(2)
                with col5:
                    st.write("courbe ROC")
                    plot_model_class(model_class,save= True)
                    st.image("AUC.png")

                with col6:
                    st.write("Rapport de classification")
                    plot_model_class(model_class,plot='class_repport',save=True)
                    st.image("Rapport class.png")
                col7,col8 = st.columns(2)
                with col7:
                    st.write("Matrice de confusion")
                    plot_model_class(model_class,plot= 'matrice de confusion',save= True)
                    st.image("Matrice de confusion.png")

                with col8:
                    st.write("Feature Importance")
                    plot_model_class(model_class,plot='feature',save =True)
                    st.image('Feature Importance.png')

            



    
if __name__=='__main__':
    main()



