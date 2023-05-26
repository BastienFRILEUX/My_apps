#Chargement des packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sns.set_palette("bright")
sns.set_context("talk")
sns.set_style("darkgrid")

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('C:\Présentation streamlit\stars.jpg') 

def main():
    st.title("Application de Machine Learning - détection de fraude bancaire")
    st.subheader("Auteur : FRILEUX Bastien")
    
    #Fonction d'importation de la base
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("Appli_fraud/creditcard.csv")
        return data
    
    #Affichage de la table de données
    df = load_data()
    df_sample = df.sample(5)
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.markdown("**Echantillon de 5 observations du jeu de données 'creditcard'**")
        st.write(df_sample)
        
    #Découpage du jeu de données
    seed = 42
    
    def split(dataframe):
        X = dataframe.drop(["Class"], axis=1)
        y = dataframe["Class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        return X_train, X_test, y_train, y_test
        
    X_train, X_test, y_train, y_test = split(df)
    
    #Choix du classifieur
    classifier = st.sidebar.selectbox("Classificateur :",
                                     ("Random Forest", "SVM", "Logistic Regression"))
    
    #Analyse de la performance du modèle
    def plot_perf(graphes):
        if len(graphes) >= 2:
            col1, col2 = st.columns(2)
            with col1: 
                if "Matrice de confusion" in graphes:
                    st.write("Matrice de confusion")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots()
                    ax_cm = sns.heatmap(cm, annot=True, fmt=".0f", cbar=False, cmap="Blues", 
                                        xticklabels = ["Pred no fraud", "Pred fraud"],
                                        yticklabels = ["No fraud", "Fraud"])
                    st.pyplot(fig_cm)

            with col2:
                if classifier != "SVM":
                    if "Courbe ROC" in graphes:
                        st.write(f"Courbe ROC - score AUC : {round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]),2)}")
                        fpr, tpr, seuil = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                        fig_cr, ax_cr = plt.subplots()
                        ax_cr = plt.plot(fpr, tpr, label="Courbe ROC")
                        plt.xlabel("Faux positifs")
                        plt.ylabel("Vrais positifs (rappel)")
                        st.pyplot(fig_cr)
                else:
                    if "Courbe ROC" in graphes:
                        st.write(f"Courbe ROC - score AUC : {round(roc_auc_score(y_test, model.decision_function(X_test)),2)}")
                        fpr, tpr, seuil = roc_curve(y_test, model.decision_function(X_test))
                        fig_cr, ax_cr = plt.subplots()
                        ax_cr = plt.plot(fpr, tpr, label="Courbe ROC")
                        plt.xlabel("Faux positifs")
                        plt.ylabel("Vrais positifs (rappel)")
                        st.pyplot(fig_cr)

            col3, col4 = st.columns(2)
            with col3:
                if classifier != "SVM":
                    if "Courbe précision-rappel" in graphes:
                        st.write(f"Précision-rappel - average precision : {round(average_precision_score(y_test, model.predict_proba(X_test)[:,1]),2)}")
                        precision, recall, seuil = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
                        fig_pr, ax_pr = plt.subplots()
                        ax_pr = plt.plot(precision, recall, label="Courbe précision-rappel")
                        plt.xlabel("Précision")
                        plt.ylabel("Rappel")
                        st.pyplot(fig_pr)
                else:
                    if "Courbe précision-rappel" in graphes:
                        st.write(f"Précision-rappel - average precision : {round(average_precision_score(y_test, model.decision_function(X_test)),2)}")
                        precision, recall, seuil = precision_recall_curve(y_test, model.decision_function(X_test))
                        fig_pr, ax_pr = plt.subplots()
                        ax_pr = plt.plot(precision, recall, label="Courbe précision-rappel")
                        plt.xlabel("Précision")
                        plt.ylabel("Rappel")
                        st.pyplot(fig_pr)
            
            if classifier == "Random Forest":
                with col4:
                    if "Features Importances" in graphes:
                        st.write("Importances des features")
                        fig_ip, ax_ip = plt.subplots()
                        feature_importances = pd.DataFrame({
                            "Features"    : X_train.columns,
                            "Importances" : model.feature_importances_}).sort_values(["Importances"], ascending=False)
                        ax_ip = sns.barplot(data=feature_importances, x="Importances", y="Features")
                        plt.xlabel("")
                        plt.ylabel("")
                        st.pyplot(fig_ip)
               
            if classifier == "Logistic Regression":
                with col4:
                    if "Coefficients" in graphes:    
                        st.write("Valeurs des coefficients")
                        fig_ip, ax_ip = plt.subplots()
                        coefficients = pd.DataFrame({
                            "Features"     : X_train.columns,
                            "coefficients" : model.coef_.reshape(-1),
                            "coef_abs"     : np.abs(model.coef_.reshape(-1))}).sort_values(["coef_abs"], ascending=False)
                        ax_ip = sns.barplot(data=coefficients, x="coefficients", y="Features")
                        plt.xlabel("")
                        plt.ylabel("")
                        st.pyplot(fig_ip)
                           
    # Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle")
        n_estimators = st.sidebar.number_input("Choisir le nombre d'arbres dans la fôret", 100, 1000, step=10)
        max_depth    = st.sidebar.number_input("Choisir la profondeur de chaque arbre", 1, X_train.shape[1], step=1)
        bootstrap    = st.sidebar.radio("Echantillons bootstrap lors de la création d'arbres ?", (True, False))
    
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle ML",
            ("Matrice de confusion", "Courbe ROC", "Courbe précision-rappel", "Features Importances"))
    
        if st.sidebar.button("Execution du modèle", key="classify"):
            st.subheader("Résultats du Random Forest")
            
            #Initialisation d'un objet RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators = n_estimators,
                max_depth    = max_depth,
                bootstrap    = bootstrap,
                random_state = seed)
            
            #Entrainement du modèle
            model.fit(X_train, y_train)
            
            #Prédiction du modèle
            y_pred = model.predict(X_test)
            
            #Métriques de performance
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            score_f1  = f1_score(y_test, y_pred)
            
            #Affichage des métriques
            st.write("Exactitude :", round(accuracy, 2))
            st.write("Précision  :", round(precision, 2))
            st.write("Rappel     :", round(recall, 2))
            st.write("Score f1   :", round(score_f1, 2))
            
            #Affichage des graphiques de performance
            plot_perf(graphes_perf)
            
    # Logistic Regression
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyperparamètres du modèle")
        hyp_C    = st.sidebar.number_input("Choisir le paramètre de régularisation C", 0.01, 100.0)
        max_iter = st.sidebar.number_input("Choisir le nombre maximum d'itérations", 100, 1000, step=10)
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle ML",
            ("Matrice de confusion", "Courbe ROC", "Courbe précision-rappel", "Coefficients"))
    
        if st.sidebar.button("Execution du modèle", key="classify"):
            st.subheader("Résultats de la régression logistique")
            
            #Initialisation d'un objet LogisticRegression
            model = LogisticRegression(
                C = hyp_C,
                max_iter = max_iter,
                random_state = seed)
            
            #Entrainement du modèle
            model.fit(X_train, y_train)
            
            #Prédiction du modèle
            y_pred = model.predict(X_test)
            
            #Métriques de performance
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            score_f1  = f1_score(y_test, y_pred)
            
            #Affichage des métriques
            st.write("Exactitude :", round(accuracy, 2))
            st.write("Précision  :", round(precision, 2))
            st.write("Rappel     :", round(recall, 2))
            st.write("Score f1   :", round(score_f1, 2))
            
            #Affichage des graphiques de performance
            plot_perf(graphes_perf)
            
    # SVM
    if classifier == "SVM":
        st.sidebar.subheader("Hyperparamètres du modèle")
        hyp_C     = st.sidebar.number_input("Choisir le paramètre de régularisation C", 0.01, 100.0)
        hyp_gamma = st.sidebar.number_input("Choisir le paramètre de régularisation gamma", 0.01, 100.0)
        kernel    = st.sidebar.radio("Choisir le Kernel", ("rbf","linear"))
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle ML",
            ("Matrice de confusion", "Courbe ROC", "Courbe précision-rappel"))
    
        if st.sidebar.button("Execution du modèle", key="classify"):
            st.subheader("Résultats de Support Vector Machine")
            
            #Initialisation d'un objet SVC
            model = SVC(
                C      = hyp_C,
                gamma  = hyp_gamma,
                kernel = kernel,
                random_state = seed)
            
            #Entrainement du modèle
            model.fit(X_train, y_train)
            
            #Prédiction du modèle
            y_pred = model.predict(X_test)
            
            #Métriques de performance
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            score_f1  = f1_score(y_test, y_pred)
            
            #Affichage des métriques
            st.write("Exactitude :", round(accuracy, 2))
            st.write("Précision  :", round(precision, 2))
            st.write("Rappel     :", round(recall, 2))
            st.write("Score f1   :", round(score_f1, 2))
            
            #Affichage des graphiques de performance
            plot_perf(graphes_perf)
    
if __name__ == '__main__':
    main()