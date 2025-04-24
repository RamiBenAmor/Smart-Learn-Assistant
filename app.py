import MySQLdb
import mysql
import os
from dataProcessSTandard import split_document_by_tokens
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from flask_mysqldb import MySQL
from dataProcess import structure_data
from quiz import generate_quiz_from_data,generate_quiz_from_context, generate_quiz_from_data1,get_first_512_words
from dataProcess import extract_text_from_pdf,structure_data,calcul_TextetMetaDataPkl
from werkzeug.security import generate_password_hash, check_password_hash
from embedding_calcul import generate_embeddings_and_faiss
from Resume import Resume_data
# Initialiser Flask
app = Flask(__name__)
global_final_data = None
app.config['SECRET_KEY'] = 'mysecretkey1234'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'gearsofwar3'
app.config['MYSQL_DB'] = 'pcd'
mysql = MySQL(app)
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
    username = request.form['username']
    email = request.form['email']
    password = generate_password_hash(request.form['password'])
    phone=request.form['phone']
    #cur = mysql.connection.cursor()
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
    existing_user = cur.fetchone()
    if existing_user:
        if existing_user['email'] == email:
          return jsonify({'success': False, 'field': 'email', 'message': "Email déjà utilisé"}), 400
        elif existing_user['username'] == username:
          return jsonify({'success': False, 'field': 'username', 'message': "username déjà utilisé"}), 400
        return redirect(url_for('page2'))
    cur.execute("INSERT INTO users (username, email, password,phone) VALUES (%s, %s, %s,%s)",
                (username, email, password,phone))
    mysql.connection.commit()
    cur.close()

    #flash("Inscription réussie 🎉", "success")
    return jsonify({'success': True}), 200
    #return redirect(url_for('home'))
# 🔐 Connexion
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    if user and check_password_hash(user[3], password):
        #session['username'] = username
        #flash("Connexion réussie ✅", "success")
        session['user_id'] = user[0]
        return redirect(url_for('interface'))
    else:
        flash("Identifiants incorrects ❌", "danger")
        return redirect(url_for('page2'))

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



@app.route('/quiz', methods=['GET'])
def quiz():
    global global_final_data 
    try:
        if global_final_data is None:
            return jsonify({"response": "Aucune donnée disponible. Veuillez d'abord effectuer l'analyse."})
        quiz_list = generate_quiz_from_data(global_final_data, 2)
        quiz_text = "\n\n".join(quiz_list)
        return jsonify({"response": quiz_text})
    except Exception as e:
        return jsonify({"response": f"Erreur lors de la génération du quiz : {str(e)}"})
    


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

"""
@app.route('/quizD', methods=['GET'])
def quiz2():
    global global_final_data 
    try:
        if global_final_data is None:
            return jsonify({"response": "Aucune donnée disponible. Veuillez d'abord effectuer l'analyse."})
        quiz_list = generate_quiz_from_data1(global_final_data, 2,3)
        quiz_text = "\n\n".join(quiz_list)
        return jsonify({"response": quiz_text})
    except Exception as e:
        return jsonify({"response": f"Erreur lors de la génération du quiz : {str(e)}"})
"""

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
