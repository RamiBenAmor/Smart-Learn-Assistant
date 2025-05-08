# authentication.py
from flask import request, jsonify,session
from werkzeug.security import generate_password_hash,check_password_hash
import MySQLdb.cursors
from extensions import mysql  # ✅ pas "from app import mysql"

def register_user():
    username = request.form['username']
    email = request.form['email']
    password = generate_password_hash(request.form['password'])
    phone = request.form['phone']

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
    existing_user = cur.fetchone()

    if existing_user:
        if existing_user['email'] == email:
            return jsonify({'success': False, 'field': 'email', 'message': "Email déjà utilisé"}), 400
        elif existing_user['username'] == username:
            return jsonify({'success': False, 'field': 'username', 'message': "username déjà utilisé"}), 400

    cur.execute("INSERT INTO users (username, email, password, phone) VALUES (%s, %s, %s, %s)",
                (username, email, password, phone))
    mysql.connection.commit()
    cur.close()

    return jsonify({'success': True}), 200

#authentication pour connexion à un compte existant
def login_user():
    email = request.form['email']
    password = request.form['password']

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    if user and check_password_hash(user[3], password):
        session['user_id'] = user[0]
        #return redirect(url_for('interface'))
        return jsonify({'success': True}), 200
    if not user:
        return jsonify({'success': False, 'field': 'email', 'message': "Aucun compte avec cet email."}), 400
    if not check_password_hash(user[3], password):
        return jsonify({'success': False, 'field': 'password', 'message': "Mot de passe incorrect."}), 400    

