##########################################################
#Page de l'application sur la modélisation               # 
##########################################################

import streamlit as st
import pandas as pd
import pandas_profiling
import joblib
from PIL import Image
from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg

from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class 

url = "https://www.linkedin.com/in/bastien-frileux-48612b250/" 

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
#add_bg_from_local('C:\Présentation streamlit\stars.jpg') 

def main():   

    selected = option_menu(
            menu_title=None,
            options=["Modélisation", "Classification", "Régression"],
            icons=["house", "star", "star"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal")
    
    if selected == "Modélisation":
    
        st.title("Mon application auto ML :sunglasses:")
        #with st.sidebar.container():
        #    image = Image.open("C:\Présentation streamlit\Bastien.png") 
        #    st.image(image, width=150)
        st.sidebar.write("[Auteur: Bastien FRILEUX](%s)" % url)
        st.sidebar.markdown(
            "**Cette application permet de produire des modèles de Machine Learning sans code**\n"
            "1. Charger votre fichier (format csv)\n"
            "2. Cliquer sur le bouton *Profil* pour générer une revue détaillée du dataset\n"
            "3. Choisir la variable cible\n"
            "4. Choisir la tâche de Machine Learning (Classification ou Régression)\n"
            "5. Cliquer sur *Run model* et visualiser les résultats\n"
            "6. Télécharger le modèle sur votre ordinateur\n"
            "7. Passer à l'onglet *Classification* ou *Régression* pour faire des prédictions"
        )

        file = st.file_uploader("**Charger votre dataset au format csv**", type=["csv"])

        if file is not None:
            data = pd.read_csv(file, sep=";")

            if st.sidebar.checkbox("Afficher les données brutes", True):
                st.write("Echantillon de 5 observations")
                st.dataframe(data.head(5))

            profile = st.button("Profil Dataset")
            if profile:
                profile_df = data.profile_report()
                st_profile_report(profile_df)

            target = st.selectbox("**Choisissez la variable cible**", data.columns)
            task = st.selectbox("**Choisissez la tâche**", ["Classification","Régression"])

            if task == "Régression":
                if st.button("Run model"):
                    s_reg = setup_reg(data, target=target, session_id=42)
                    model_reg = compare_models_reg()
                    save_model_reg(model_reg, "best_reg_model")
                    st.success("Le modèle de régression a été contruit avec succès")

                    #Affichage des résultats
                    st.write("Residuals")
                    plot_model_reg(model_reg, plot="residuals", save=True)
                    st.image("Residuals.png")

                    st.write("Feature importance")
                    plot_model_reg(model_reg, plot="feature", save=True)
                    st.image("Feature Importance.png")

                    with open("best_reg_model.pkl", "rb") as f:
                        st.download_button("Télécharger le meilleur modèle", f, file_name="best_reg_model.pkl")

            if task == "Classification":
                if st.button("Run model"):
                    s_class = setup_class(data, target=target, session_id=42)
                    model_class = compare_models_class(include=["et","lr","rf","gbc"])
                    save_model_class(model_class, "best_class_model")
                    st.success("Le modèle de classification a été contruit avec succès")

                    #Affichage des résultats
                    col5, col6 = st.columns(2)
                    with col5:
                        st.write("Courbe ROC")
                        plot_model_class(model_class, plot="auc", save=True)
                        st.image("AUC.png")

                    with col6:
                        st.write("Rapport de classification")
                        plot_model_class(model_class, plot="class_report", save=True)
                        st.image("Class Report.png")

                    col7, col8 = st.columns(2)
                    with col7:
                        st.write("Matrice de confusion")
                        plot_model_class(model_class, plot="confusion_matrix", save=True)
                        st.image("Confusion Matrix.png")

                    with col8:
                        st.write("Feature importance")
                        plot_model_class(model_class, plot="feature", save=True)
                        st.image("Feature Importance.png")

                    #Téléchargement du modèle
                    with open("best_class_model.pkl", "rb") as f:
                        st.download_button("Télécharger le meilleur modèle", f, file_name="best_class_model.pkl")
                        
    if selected == "Classification":
        
        #add_bg_from_local('C:\Présentation streamlit\iris2.jpg') 
    
        st.title("Prédictions de l'espèce d'Iris")
        
        st.sidebar.write("[Auteur: Bastien FRILEUX](%s)" % url)
        st.sidebar.header("Les paramètres du modèle")

        sepal_lenght = st.sidebar.slider("Longueur du sépale", 4.3, 7.9, 5.3)
        sepal_width  = st.sidebar.slider("Largeur du sépale", 2.0, 4.4, 3.3)
        petal_lenght = st.sidebar.slider("Longueur du pétale", 1.0, 6.9, 2.3)
        petal_width  = st.sidebar.slider("Largeur du pétale", 0.1, 2.5, 1.3)

        data={"SepalLengthCm" : sepal_lenght,
              "SepalWidthCm"  : sepal_width,
              "PetalLengthCm"  : sepal_width,
              "PetalWidthCm"  : sepal_width}

        input_parametres_class=pd.DataFrame(data, index=[0])

        st.subheader("**Prédisons la classe pour ces paramètres :**")
        st.write(input_parametres_class)

        loaded_model = joblib.load("best_class_model.pkl")
        y_pred_class = loaded_model.predict(input_parametres_class)
        st.subheader(f"**Correspond à :blue[{y_pred_class[0]}]**")
    
    if selected == "Régression":
         
        #add_bg_from_local('C:\Présentation streamlit\image_car2.png')  

        st.title("Prédictions du prix d'une voiture")
        
        st.sidebar.write("[Auteur: Bastien FRILEUX](%s)" % url)
        st.sidebar.header("Les paramètres du modèle")

        #Chargement de la base utlisée
        @st.cache_data(persist=True) #Si le programme tourne plusieurs fois et que la base n'a pas changé, alors ça ne recharge pas.
        def load_data():
            data = pd.read_csv("cars_predict.csv", sep=";")
            return data

        cars = load_data()

        #Variables catégorielles à sélectionner
        CompanyName_select    = cars["CompanyName"].unique()
        fueltype_select       = cars["fueltype"].unique()
        aspiration_select     = cars["aspiration"].unique()
        doornumber_select     = cars["doornumber"].unique()
        carbody_select        = cars["carbody"].unique()
        drivewheel_select     = cars["drivewheel"].unique()
        enginelocation_select = cars["enginelocation"].unique()
        enginetype_select     = cars["enginetype"].unique()
        cylindernumber_select = cars["cylindernumber"].unique()
        fuelsystem_select     = cars["fuelsystem"].unique()

        CompanyName    = st.sidebar.selectbox("Choisir le nom de la compagnie", CompanyName_select)
        fueltype       = st.sidebar.selectbox("Choisir type carburant", fueltype_select)
        aspiration     = st.sidebar.selectbox("Choisir aspiration", aspiration_select)
        doornumber     = st.sidebar.selectbox("Choisir nombre de portes", doornumber_select)
        carbody        = st.sidebar.selectbox("Choisir carroserie", carbody_select)
        drivewheel     = st.sidebar.selectbox("Choisir roue motrice", drivewheel_select)
        enginelocation = st.sidebar.selectbox("Choisir emplacement du moteur", enginelocation_select)
        enginetype     = st.sidebar.selectbox("Choisir type de moteur", enginetype_select)
        cylindernumber = st.sidebar.selectbox("Choisir nombre de cylindre", cylindernumber_select)
        fuelsystem     = st.sidebar.selectbox("Choisir fuel system", fuelsystem_select)

        #Variables continues  
        wheelbase        = st.sidebar.slider("Valeur de wheelbase", cars["wheelbase"].min(), cars["wheelbase"].max(), 100.0)
        carlength        = st.sidebar.slider("Valeur de longueur de la voiture", cars["carlength"].min(), cars["carlength"].max(), 170.0)
        carwidth         = st.sidebar.slider("Valeur de largeur de la voiture", cars["carwidth"].min(), cars["carwidth"].max(), 70.0)
        carheight        = st.sidebar.slider("Valeur de hauteur de la voiture", cars["carheight"].min(), cars["carheight"].max(), 53.0)
        curbweight       = st.sidebar.slider("Valeur de poids à vide", cars["curbweight"].min(), cars["curbweight"].max(), 2500)
        enginesize       = st.sidebar.slider("Valeur de taille du moteur", cars["enginesize"].min(), cars["enginesize"].max(), 200)   
        boreratio        = st.sidebar.slider("Valeur de rapport d'alésage", cars["boreratio"].min(), cars["boreratio"].max(), 3.0)
        stroke           = st.sidebar.slider("Valeur de stroke", cars["stroke"].min(), cars["stroke"].max(), 3.0)
        compressionratio = st.sidebar.slider("Valeur de ratio de compression", cars["compressionratio"].min(), cars["compressionratio"].max(), 15.0)
        horsepower       = st.sidebar.slider("Valeur de puissance", cars["horsepower"].min(), cars["horsepower"].max(), 150)
        peakrpm          = st.sidebar.slider("Valeur de vitesse de pointe", cars["peakrpm"].min(), cars["peakrpm"].max(), 5500)

        data={'fueltype': fueltype,
              'aspiration': aspiration,
              'CompanyName': CompanyName,
              'doornumber': doornumber,
              'carbody': carbody,
              'drivewheel': drivewheel,
              'enginelocation': enginelocation,
              'wheelbase': wheelbase,
              'carlength': carlength,
              'carwidth': carwidth,
              'carheight': carheight,
              'curbweight': curbweight,
              'enginetype': enginetype,
              'cylindernumber': cylindernumber,
              'enginesize': enginesize,
              'fuelsystem': fuelsystem,
              'boreratio': boreratio,
              'stroke': stroke,
              'compressionratio': compressionratio,
              'horsepower': horsepower,
              'peakrpm': peakrpm}

        input_parametres=pd.DataFrame(data, index=[0])

        st.subheader("**Prédisons le prix pour ces paramètres :**")
        st.write(input_parametres)

        loaded_model = joblib.load("best_reg_model.pkl")
        y_pred = loaded_model.predict(input_parametres)
        st.subheader(f'**Le prix de cette voiture est estimé à :blue[{round(y_pred[0],2)}$]**')
        
if __name__ == '__main__':
    main()
        
        