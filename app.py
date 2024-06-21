import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class

import os
import time

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

def main():
    st.title("YaoML")
    st.sidebar.write("[Auteur: Yao Jean Bosco]")
    st.sidebar.markdown(
        "**Cette application web est un outil NO-Code pour l'analyse exploratoire de données et la construction de modèles d'apprentissage automatique pour les tâches de régression et de classification*** \n\n"

            "1.Chargez votre jeux de données (fichier csv);\n"

            "2.Cliquez sur le bouton Executer pour générer le profilage pandas du jeu de données;\n\n"
            "3.Choisissez votre variable cible;\n"

            "4.Choisissez la tâche d'apprentissage automatique (régression ou classification);\n\n"

            "5.Cliquez sur **Exécuter **la modélisation pour démarrer le processus d'entraînement. Lorsque le modèle est construit, vous pouvez voir les résultats comme le modèle de pipeline,le graphique des résidus, la courbe ROC, la matrice de confusion, l'importance des caractéristiques, etc.\n\n"

            "6.Téléchargez le modèle de pipeline sur votre ordinateur local.")
    
    file = st.file_uploader("Charger votre fichier csv ici:", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.dataframe(data.head())

        profile = st.button("Rapport Dataset")
        if profile:
            profile_df = data.profile_report()
            st_profile_report(profile_df)

        VarCible = st.selectbox("Variable cible", data.columns)
        tache = st.selectbox("Choisir action", ["Regression", "Classification"])
        data = data.dropna(subset=VarCible) # pour que mon appli fonctionne même s'il y'a des valeur manquante
        if st.button("Executer"):
            if tache == "Regression":
                exo_reg = setup_reg(data, target=VarCible)
                model_reg = compare_models_reg()
                save_model_reg(model_reg, "meilleur_modèle_regression")
                st.success("Le modèle de regression a été construit avec succès!")

                # Affichage des resultats
                st.write("Residus")
                plot_model_reg(model_reg, plot='residuals', save=True, verbose=False)
                residuals_path = "Residuals.png"
                time.sleep(1)  # delais pour s'assurer que l'image a été sauvegardé
                if os.path.exists(residuals_path):
                    st.image(residuals_path)
                
                st.write("Feature Importance")
                plot_model_reg(model_reg, plot='feature', save=True, verbose=False)
                feature_importance_path = "Feature_Importance.png"
                time.sleep(1)  # delais pour s'assurer que l'image a été sauvegardé
                if os.path.exists(feature_importance_path):
                    st.image(feature_importance_path)

                if os.path.exists('meilleur_modèle_regression.pkl'):
                    with open('meilleur_modèle_regression.pkl', 'rb') as f:
                        st.download_button('Télécharger le modèle', f, file_name="meilleur_modèle_regression.pkl")
                else:
                    st.error("Le fichier du modèle n'a pas été trouvé.")

            elif tache == "Classification":
                exp_class = setup_class(data, target=VarCible)
                model_class = compare_models_class()
                save_model_class(model_class, 'meilleur_modèle_classification')
                st.success("Modèle de classification exécuté avec succès!")

                # Affichage des resultats
                col5, col6 = st.columns(2)
                with col5:
                    st.write("Courbe ROC")
                    plot_model_class(model_class, plot='auc', save=True, verbose=False)
                    auc_path = "AUC.png"
                    time.sleep(1) # delais pour s'assurer que l'image a été sauvegardé
                    if os.path.exists(auc_path):
                        st.image(auc_path)

                with col6:
                    st.write("Rapport de classification")
                    plot_model_class(model_class, plot='class_report', save=True, verbose=False)
                    class_report_path = "Class_Report.png"
                    time.sleep(1)  
                    if os.path.exists(class_report_path):
                        st.image(class_report_path)

                col7, col8 = st.columns(2)
                with col7:
                    st.write("Matrice de confusion")
                    plot_model_class(model_class, plot='confusion_matrix', save=True, verbose=False)
                    confusion_matrix_path = "Confusion_Matrix.png"
                    time.sleep(1)  
                    if os.path.exists(confusion_matrix_path):
                        st.image(confusion_matrix_path)

                with col8:
                    st.write("Feature Importance")
                    plot_model_class(model_class, plot='feature', save=True, verbose=False)
                    feature_importance_path = "Feature_Importance.png"
                    time.sleep(1)  # Adding a delay to ensure the file is saved
                    if os.path.exists(feature_importance_path):
                        st.image(feature_importance_path)
                    else:
                        st.error("Feature importance plot not found.")

                if os.path.exists('meilleur_modèle_classification.pkl'):
                    with open('meilleur_modèle_classification.pkl', 'rb') as f:
                        st.download_button('Télécharger le modèle', f, file_name="meilleur_modèle_classification.pkl")
                else:
                    st.error("Le fichier du modèle n'a pas été trouvé.")
    else:
        st.image("D:\Data Science\semestre8\pythonAvancé\projetPythonAvancé\home_image.png")

if __name__ == '__main__':
    main()
