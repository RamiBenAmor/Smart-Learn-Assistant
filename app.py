from extensions import mysql 
import os
from dataProcessSTandard import split_document_by_tokens
from flask import Flask, render_template, request, redirect, url_for,jsonify
from dataProcess import structure_data
from quiz import generate_quiz_from_data1
from dataProcess import extract_text_from_pdf,structure_data,calcul_TextetMetaDataPkl
from embedding_calcul import generate_embeddings_and_faiss
from Resume import Resume_data
from authentication import register_user,login_user
# Initialiser Flask
app = Flask(__name__)
global_final_data = None
app.config['SECRET_KEY'] = 'mysecretkey1234'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'gearsofwar3'
app.config['MYSQL_DB'] = 'pcd'
mysql.init_app(app)  # ✅ initialisation ici
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def home():
    return render_template('page1.html')
@app.route('/page2')
def page2():
    return render_template('accueil.html')
@app.route('/interface')
def interface():
    return render_template('interface.html')


@app.route('/index')
def index():
    return render_template('index.html')
# 📝 Inscription
@app.route('/register', methods=['POST'])
def register():
    return register_user()
# 🔐 Connexion
@app.route('/login', methods=['POST'])
def login():
    return login_user()
# Route de chat
@app.route('/chat', methods=['POST'])
def chat():
    from Main import get_contextual_response,retrieve_relevant_texts
    user_input = request.json.get("message")
    context = retrieve_relevant_texts(user_input)
    response = get_contextual_response(context,user_input)
    #response="bla bla bla"
    
    # Retourner la réponse sous forme de JSON
    return jsonify({"response": response})

"""Apres click sur bouton Recherche Approfondie"""
@app.route('/RechercheApprofondie', methods=['POST'])
def RechercheApprofondie():
    global global_final_data 
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'message': 'Aucun fichier envoyé.'}), 400

    file = request.files['file']
    
    # Vérifier si le fichier a un nom (n'est pas vide)
    if file.filename == '':
        return jsonify({'message': 'Aucun fichier sélectionné.'}), 400

    try:
        # Sauvegarder le fichier sur le serveur
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_path = extract_text_from_pdf(filepath, app.config['UPLOAD_FOLDER'])
        final_data = structure_data(output_path)
        calcul_TextetMetaDataPkl(final_data)

        # Appeler la fonction pour générer les embeddings et l'index FAISS
        generate_embeddings_and_faiss()
        global_final_data = final_data 
        # Retourner une réponse après l'exécution de l'analyse
        return redirect(url_for('index'))


    
    except Exception as e:
        # Gérer les erreurs éventuelles
        return jsonify({'message': f'Erreur lors de l\'analyse : {str(e)}'}), 500
    



@app.route('/RechercheStandard', methods=['POST'])
def RechercheStandard():
    global global_final_data 
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'message': 'Aucun fichier envoyé.'}), 400

    file = request.files['file']
    
    # Vérifier si le fichier a un nom (n'est pas vide)
    if file.filename == '':
        return jsonify({'message': 'Aucun fichier sélectionné.'}), 400

    try:
        # Sauvegarder le fichier sur le serveur
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_path = extract_text_from_pdf(filepath, app.config['UPLOAD_FOLDER'])
        final_data =split_document_by_tokens(output_path)
        
        calcul_TextetMetaDataPkl(final_data)
        # Appeler la fonction pour générer les embeddings et l'index FAISS
        generate_embeddings_and_faiss()
        global_final_data = final_data 
        # Retourner une réponse après l'exécution de l'analyse
        return redirect(url_for('index'))
    except Exception as e:
        # Gérer les erreurs éventuelles
        return jsonify({'message': f'Erreur lors de l\'analyse : {str(e)}'}), 500

@app.route('/quizS', methods=['GET'])
def quiz1():
    global global_final_data 
    try:
        if global_final_data is None:
            return jsonify({"response": "Aucune donnée disponible. Veuillez d'abord effectuer l'analyse."})
        quiz_list = generate_quiz_from_data1(global_final_data, 2,1)
        quiz_text = "\n\n".join(quiz_list)
        return jsonify({"response": quiz_text})
    except Exception as e:
        return jsonify({"response": f"Erreur lors de la génération du quiz : {str(e)}"})


@app.route('/quizH', methods=['GET'])
def quizH():
    global global_final_data 
    try:
        if global_final_data is None:
            return jsonify({"response": "Aucune donnée disponible. Veuillez d'abord effectuer l'analyse."})
        quiz_list = generate_quiz_from_data1(global_final_data, 2,3)
        quiz_text = "\n\n".join(quiz_list)
        return jsonify({"response": quiz_text})
    except Exception as e:
        return jsonify({"response": f"Erreur lors de la génération du quiz : {str(e)}"})

@app.route('/quizM', methods=['GET'])
def quizM():
    global global_final_data 
    try:
        if global_final_data is None:
            return jsonify({"response": "Aucune donnée disponible. Veuillez d'abord effectuer l'analyse."})
        quiz_list = generate_quiz_from_data1(global_final_data, 2,2)
        quiz_text = "\n\n".join(quiz_list)
        return jsonify({"response": quiz_text})
    except Exception as e:
        return jsonify({"response": f"Erreur lors de la génération du quiz : {str(e)}"})



@app.route('/Resume',methods=['GET'])
def Resume():
    global global_final_data 
    try:
        if global_final_data is None:
            return jsonify({"response": "Aucune donnée disponible. Veuillez d'abord effectuer l'analyse approfondie."})
        print("en cours ")
        resume_list = Resume_data(global_final_data)
        resume_text = "\n\n".join(resume_list)
        return jsonify({"response": resume_text})
    except Exception as e:
        return jsonify({"response": f"Erreur lors de la génération du quiz : {str(e)}"})
    






#Dans la route /login, tu envoies un formulaire classique (form POST) depuis une page HTML.
#🔹 Dans la route /RechercheApprofondie, tu envoies une requête AJAX / fetch() via JavaScript.
    # Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=False)
